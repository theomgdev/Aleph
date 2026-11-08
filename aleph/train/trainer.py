"""Training loop for a core whose depth is the sequence.

Three things here are specific to this architecture rather than borrowed from a
transformer recipe.

**Truncated BPTT, inside the chunk and across calls.** The state crosses every
boundary detached, so information flows forward while gradient does not. This
is not a memory optimization here but what makes the model trainable: the
per-step Jacobian of a shared matrix sits slightly above 1 (measured 1.0647 at
N=512), so backpropagating a long sequence whole is exponential -- gradient
norm 523 at 128 steps against 151,091 at 256, and identical with the refinement
branches switched off, so it is the recurrence rather than any added mechanism.
`window` bounds the backward length; every position is still predicted and
still supervised, and because `W` is one shared matrix what is learned inside a
window applies at every step.

**Warmup is not optional.** The refinement branches add a second gradient path
into each step, and the usable learning rate is correspondingly narrower:
measured on a 64-sequence overfit, `depth=4` reached 0.0192 nats at 1e-3 and
diverged at 3e-3, while `depth=0` tolerated both. A linear warmup into a cosine
decay makes the early steps small enough that the branches settle before the
rate peaks, which is what lets a depth comparison measure depth.

**Telemetry over adjectives.** Every log line carries the numbers that
distinguish the failure modes this architecture actually has: `bpb` for a
tokenizer-independent score, `epochs` to tell data starvation from
over-parameterization, `|g|` before clipping, `clip%` for how often the clip
binds -- a clip that binds every step has replaced the schedule with `1/|g|`,
which is a finding and not a detail -- and `|h|` to confirm the state is on its
`sqrt(N)` sphere. A run that goes wrong goes wrong in one of those columns.
"""

import math
import time

import torch
import torch.nn.functional as F

from .muon import MuonAdamW

# TF32 on the tensor cores: the matmuls here are small and repeated `T` times,
# so this is the precision/throughput point that measured best, and setting it
# once at import beats every caller remembering to.
torch.set_float32_matmul_precision('high')


class Trainer:
    """Train an `AlephCore` on a `TokenStream`.

    Args:
        model (AlephCore): The model.
        lr (float): Peak learning rate after warmup. Default 1e-3 -- the rate
            that converged at `depth=4` where 3e-3 diverged.
        warmup (int): Linear warmup steps. Default 200.
        total (int, optional): Steps for the cosine decay horizon. Defaults to
            whatever `run()` is given, so the schedule matches the actual run.
        min_lr_frac (float): Floor of the cosine decay, as a fraction of `lr`.
            Default 0.1.
        weight_decay (float): Decoupled, applied to `W`, `U` and `V` only.
            Norm gains, gates, biases and the codebook's scalars are excluded:
            their zero is not a neutral point, and decaying the codebook's
            unigram bias would discard a free 3.9 bits.
        clip (float): Gradient-norm clip. Default 1.0. Watch `clip%` in the
            log: binding on every step means the clip, not the schedule, is
            setting the step size.
        window (int): Backward window in steps. Default 32. The lever that
            makes a long sequence trainable at all -- see the module docstring.
            0 backpropagates the whole chunk, which is exponential past ~128
            steps.
        device: Torch device.
        optimizer (str): `'muon'` or `'adamw'`. Default `'muon'`: at identical
            parameters and seed, orthogonalized updates on the 2D operators
            measured 4.7643 val nats against AdamW's 5.2949 at 400 steps --
            the same score AdamW needed 900 steps to reach.
        reset_every (int): Start a fresh sequence every N steps, state and all
            caches. Default 16. Carrying state indefinitely is what makes the
            short-chunk geometry free, but it also means a cold start is ~0.02%
            of training positions while being 100% of generation's first token
            -- measured 11.54 nats at position 0 against 2.74 warm, and 38
            tokens to recover. The model has to practise beginning.
        ponder_beta (float): Weight on the halting KL when the model ponders.
            Nothing in a cross-entropy prefers a short ponder and a mixture of
            states is free capacity, so without this the head spends the
            maximum on every token -- fixed `wait` reached by gradient descent.
            Ignored when the model has no halting head.
    """

    def __init__(self, model, lr=1e-3, warmup=200, total=None, min_lr_frac=0.1,
                 weight_decay=0.01, clip=1.0, window=32, device=None,
                 optimizer='muon', reset_every=16, ponder_beta=0.01):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.lr, self.warmup, self.total = lr, warmup, total
        self.min_lr_frac, self.clip, self.window = min_lr_frac, clip, window
        self.reset_every = reset_every
        self.ponder_beta = ponder_beta
        self.step_count = 0

        if optimizer == 'muon':
            self.opt = MuonAdamW.from_model(model, lr=lr, weight_decay=weight_decay)
        elif optimizer == 'adamw':
            decay, no_decay = [], []
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                # Connective structure decays; everything that modulates does
                # not. The sigmoid-gated logits in particular have no neutral
                # zero, so decay would silently close them.
                is_operator = p.ndim == 2 and not name.endswith('beta')
                (decay if is_operator else no_decay).append(p)
            self.opt = torch.optim.AdamW([
                {'params': decay, 'weight_decay': weight_decay},
                {'params': no_decay, 'weight_decay': 0.0},
            ], lr=lr, betas=(0.9, 0.95), eps=1e-8)
        else:
            raise ValueError(f"optimizer must be 'muon' or 'adamw', got {optimizer!r}")

        self.clipped = 0
        self.since_log = 0

    def lr_at(self, step):
        """Linear warmup into cosine decay."""
        if step < self.warmup:
            return self.lr * (step + 1) / self.warmup
        total = self.total or max(self.warmup * 10, step + 1)
        t = min(1.0, (step - self.warmup) / max(1, total - self.warmup))
        return self.lr * (self.min_lr_frac
                          + (1 - self.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * t)))

    def train_step(self, x, y, state=None):
        """One optimizer step. Returns `(metrics, detached_state)`."""
        self.model.train()
        lr = self.lr_at(self.step_count)
        if isinstance(self.opt, MuonAdamW):
            self.opt.set_lr(lr)
        else:
            for g in self.opt.param_groups:
                g['lr'] = lr

        # No autocast. The step applies the same few matrices `T` times, so
        # casting them per step costs `3*T` extra kernels on a loop that is
        # already launch-bound -- measured 31% slower than running the matmuls
        # in TF32, which is also the more precise of the two.
        logits, h = self.model(x, state=state, window=self.window)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), y.reshape(-1))
        ce = loss.item()

        # The KL is the only term that prices time: cross-entropy is happy to
        # spend the maximum, so the reported loss stays the cross-entropy and
        # the regularizer is added only to what is differentiated.
        ponder = getattr(self.model, 'ponder', None)
        steps = 0.0
        if ponder is not None and ponder.kl is not None:
            loss = loss + self.ponder_beta * ponder.kl
            steps = ponder.mean_steps

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        if gnorm > self.clip:
            self.clipped += 1
            self.since_log += 1
        self.opt.step()
        self.step_count += 1

        return {
            'loss': ce, 'lr': lr, 'gnorm': gnorm.item(),
            'hnorm': h.detach().norm(dim=-1).mean().item(), 'steps': steps,
        }, h.detach()

    @torch.no_grad()
    def evaluate(self, stream, batches=20):
        """Held-out loss and bits per byte.

        bits/byte rather than perplexity: it is the one score comparable across
        tokenizers, since it divides by the *text* the tokens encode rather than
        by their count. Scoped to this cache -- a rebuilt cache moves the split.
        """
        self.model.eval()
        total, count, state = 0.0, 0, None
        for _ in range(batches):
            x, y = stream.next()
            x = torch.as_tensor(x, device=self.device)
            y = torch.as_tensor(y, device=self.device)
            logits, state = self.model(x, state=state)
            total += F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                                     y.reshape(-1)).item()
            count += 1
        loss = total / max(count, 1)
        bpt = loss / math.log(2)
        return {'loss': loss, 'bits_per_token': bpt,
                'bits_per_byte': bpt / stream.meta['bytes_per_token'],
                'perplexity': math.exp(min(loss, 30))}

    def run(self, train_stream, val_stream, steps, log_every=50, eval_every=500,
            eval_batches=20, on_eval=None):
        """Train for `steps`, logging a line an ML scientist can act on."""
        if self.total is None:
            self.total = steps
        state, best, t0 = None, float('inf'), time.perf_counter()
        window, tokens = [], 0
        ponders = getattr(self.model, 'ponder', None) is not None

        header = (f"{'step':>6} {'loss':>7} {'bpb':>6} {'lr':>9} {'|g|':>6} "
                  f"{'clip%':>6} {'|h|':>6}" + (f" {'wait':>5}" if ponders else '')
                  + f" {'ep':>5} {'tok/s':>8}")
        print(header)
        print('-' * len(header))

        for step in range(steps):
            x, y = train_stream.next()
            x = torch.as_tensor(x, device=self.device)
            y = torch.as_tensor(y, device=self.device)
            # Periodically begin again, so the first few positions of a
            # sequence get gradient at all. `state=None` resets the branch
            # caches too.
            if self.reset_every and step % self.reset_every == 0:
                state = None
            m, state = self.train_step(x, y, state)
            window.append(m)
            tokens += x.numel()

            if (step + 1) % log_every == 0:
                avg = sum(w['loss'] for w in window) / len(window)
                bpb = avg / math.log(2) / train_stream.meta['bytes_per_token']
                wait_col = (f" {sum(w['steps'] for w in window) / len(window):>5.2f}"
                            if ponders else '')
                print(f"{step + 1:>6} {avg:>7.4f} {bpb:>6.3f} {m['lr']:>9.2e} "
                      f"{sum(w['gnorm'] for w in window) / len(window):>6.2f} "
                      f"{self.since_log / len(window) * 100:>5.0f}% "
                      f"{m['hnorm']:>6.2f}{wait_col} {train_stream.epochs:>5.2f} "
                      f"{tokens / (time.perf_counter() - t0):>8,.0f}", flush=True)
                window = []
                self.since_log = 0

            if eval_every and (step + 1) % eval_every == 0:
                ev = self.evaluate(val_stream, eval_batches)
                flag = ' *best' if ev['loss'] < best else ''
                best = min(best, ev['loss'])
                print(f"       val loss {ev['loss']:.4f}  bpb {ev['bits_per_byte']:.4f}  "
                      f"ppl {ev['perplexity']:.2f}{flag}", flush=True)
                if on_eval:
                    on_eval(step + 1, ev, ev['loss'] <= best)

        return {'steps': steps, 'best_val_loss': best, 'tokens': tokens,
                'seconds': time.perf_counter() - t0,
                'epochs': train_stream.epochs}
