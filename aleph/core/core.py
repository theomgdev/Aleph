"""The Aleph core: one square matrix, read as a language model.

The state `h` is `(B, N)` and one token is one step:

    s = h @ W + codebook.inject(x_t) + b
    h = RMSNorm(phi(s) + depth_refine(s))

`W` is the whole model. There are no layers, so there is nothing to stack and
nothing to share -- parameter reuse is not a trick applied to the architecture,
it is what the architecture is.

Why one step per token
----------------------
OdyssNet echoes `T` times per token, which puts `T` sequential launches between
consecutive tokens and makes the forward pass launch-bound: measured on a 3060
Ti, 64x the batch cost 1.7x the time, so the clock was tracking launches rather
than arithmetic. Depth per token is therefore bought *inside* the step instead,
where it is a wider matmul rather than another launch, and the sequence axis
gets exactly one step per token.

`depth` low-rank refinements
----------------------------
A single `N x N` application is one linear map and a nonlinearity. Capacity
beyond that comes from `depth` low-rank branches read off the same signal and
summed back into it:

    out = phi(s) + sum_k phi(RMSNorm(phi(s)) @ U_k) @ V_k

`U_k, V_k` are `N x rank`, so `depth` branches cost `2 * depth * N * rank`
against the core's `N^2` -- at `rank = N/8` and `depth = 4` that is the core
again, for four more nonlinear maps.

The branches are *parallel*, not chained, and that is a throughput decision
rather than an expressiveness one. This loop is launch-bound: measured on a
3060 Ti, 128x the batch cost 1.15x the time, so the clock tracks kernel
launches and arithmetic inside an existing launch is nearly free. A chained
stack (`r_{k+1} = r_k + ...`) needs `3 * depth` sequential launches per token
and measured 4.59x slower at `depth=4` for 2.0x the parameters -- it bought
capacity at a worse rate than it cost time. Parallel branches concatenate into
one `(N, depth*rank)` matmul and one `(depth*rank, N)` matmul, so `depth` costs
two launches whatever its value, and the extra arithmetic lands in the regime
where arithmetic is free.

Two properties of the form are load-bearing. A shared pre-norm bounds what
every branch reads while the residual it adds to stays unnormalized, so the
signal survives the sum. And `V_k` is zero-initialized, so every branch starts
as the identity: `depth` is an ablation with one variable, the same seed gives
the same `W` at any depth, and a deeper model begins its life numerically
identical to a shallower one.

What depth costs is step-size tolerance. The branches add a second gradient
path into the same step, so the usable learning rate is narrower than a
`depth=0` core's -- measured on a 64-sequence overfit, `depth=4` reached 0.0192
nats at 1e-3 and diverged at 3e-3 while `depth=0` tolerated both. That is why
the trainer warms up, and why a depth comparison at one fixed rate measures the
schedule instead of the mechanism.

Invariants
----------
* `W.diagonal() == 0`. A neuron's self-connection lives in `self_gain`, where
  it is one parameter per neuron rather than a row of `W`, and it is gated shut
  at initialization so the core has to earn the latch.
* Per-step RMSNorm pins `|h|` to `sqrt(N)` exactly, so the forward pass cannot
  explode or vanish whatever `W`'s spectral radius -- no spectral constraint,
  clamp or learnable decay appears anywhere here. Measured 15.999998 at N=256.
* The *backward* pass has no such bound, and this is the one place the
  architecture needs care. RMSNorm fixes the state's norm but its Jacobian still
  passes gradient, and the per-step Jacobian of a shared matrix sits slightly
  above 1 (measured 1.0647 at N=512, untrained): compounded over a long sequence
  that is exponential, and the gradient norm was measured at 523 for 128 steps
  against 151,091 for 256 -- 289x per doubling, and identical at `depth=0`, so it
  is the recurrence rather than the branches. `window` is therefore not a tuning
  knob but the mechanism that makes the model trainable at all: credit is
  assigned over `window` steps and the state crosses each boundary detached, so
  information still flows the whole length of the sequence while gradient does
  not. Because `W` is one shared matrix, what is learned inside a window applies
  at every step, and a window only has to span the dependencies being learned.
* `V` zero-init and `self_gain` zero-init make every optional mechanism start
  inert, so an ablation arm differs from the baseline only in what it was asked
  to differ in -- at the cost that `U`, `r_norm` and `self_logit` take no
  gradient at step 0, by design, and open once the tensor in front of them moves.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from .attention import TemporalAttention
from .codebook import Codebook
from .plasticity import Plasticity


def quantize_ste(w, mode):
    """Fake-quantize `w` with a straight-through estimator, per output row.

    QAT rather than post-training quantization, because the same matrix is
    applied at every step: a perturbation that a feedforward layer absorbs once
    gets re-applied `T` times here, and the only way the core can learn stable
    dynamics on the discrete operator is to see it during training.

    Measured on the 1.03M-parameter synthesis at 400 steps (val nats, from
    4.4243 unquantized): int8 4.4383, int4 4.6913, ternary 4.9016. So int8 is
    free and below that the core pays real capacity -- consistent with the
    published finding that ternary needs scale this model does not have.
    """
    if mode in (None, 'none'):
        return w
    if mode == 'ternary':
        # BitNet b1.58: absmean scale, weights in {-1, 0, +1}.
        scale = w.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)
        q = (w / scale).round().clamp(-1, 1) * scale
    elif mode in ('int8', 'int4'):
        qmax = 127 if mode == 'int8' else 7
        scale = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
        q = (w / scale).round().clamp(-qmax, qmax) * scale
    else:
        raise ValueError(f"quant must be None, 'none', 'int8', 'int4' or 'ternary', got {mode!r}")
    # Forward is exactly `q`, backward is exactly the identity. Both halves
    # need care: `w - w.detach()` is exactly zero where `w + (q - w).detach()`
    # leaves a one-ulp residue, so an exported weight really is ternary; and
    # `q` must be detached because `scale` is a function of `w`, so leaving it
    # live would send gradient through the quantizer's own scale.
    return q.detach() + (w - w.detach())


class AlephCore(nn.Module):
    """A square recurrent core that reads and writes tokens through frozen codes.

    Args:
        neurons (int): State width `N`. The core is `N x N`.
        codebook (Codebook): The token interface. Its `width` must equal
            `neurons` -- codes live in the state's own space, so there is no
            projection between them to pay for.
        depth (int): Low-rank branches per step. 0 disables them. Default 4.
            Costs two kernel launches whatever its value, so it is the cheap
            axis in a launch-bound loop.
        rank (int, optional): Refinement rank. Defaults to `max(4, N // 8)`.
        activation (str): `'gelu'`, `'silu'`, `'tanh'` or `'relu'`. Default
            `'gelu'`.
        self_gain (bool): Per-neuron self-connection, gated shut at init.
            Default True.
        attn (bool): Temporal attention over the state history. Default True.
            Measured -0.0644 val nats at +164k params (head_dim 32); the
            default head_dim of 16 halves that cost for the same score.
        attn_heads, attn_kv_heads, attn_head_dim, attn_window: forwarded to
            `TemporalAttention`.
        plastic (bool): Delta-rule fast weights. Default True. Measured
            -0.1191 val nats at +169k params (dim 64); the default dim of 32
            quarters that cost for the same score.
        plastic_dim (int): Fast-weight width `D`; the state is `D x D` per
            stream. Default 32.
        aux (bool): Auxiliary encoder and readout adapter. Default True, and
            the best value per parameter measured: -0.2569 val nats for +68k.
            A causal depthwise conv over injected codes gives the core local
            n-gram context before the recurrence starts, and a low-rank adapter
            rotates the state into the frozen codebook's frame so `h` does not
            have to serve as both recurrent memory and readout vector.
        aux_rank (int): Rank of the readout adapter. Default 64.
        aux_kernel (int): Width of the causal conv over injected codes, so the
            n in n-gram. Default 4.
        quant (str, optional): `'int8'`, `'int4'` or `'ternary'` QAT on the
            recurrent operators. Default None. int8 measured free; see
            `quantize_ste`.
        wait (int): Extra core steps per token, on a frozen wait code, with no
            readout. 0 disables them. Depth here is time, so this buys
            deliberation per token the way `depth` buys it per step -- but
            unlike `depth` it costs sequential launches, which is the expensive
            axis in this loop.
        dropout (float): Applied to the state after the activation. Default 0.
        gradient_checkpointing (bool): Recompute steps in the backward pass.
            Trades ~30% time for a step-count-independent memory profile.
        device: Torch device.
    """

    def __init__(self, neurons, codebook, depth=4, rank=None, activation='gelu',
                 self_gain=True, attn=True, attn_heads=4, attn_kv_heads=1,
                 attn_head_dim=16, attn_window=128, plastic=True, plastic_dim=32,
                 aux=True, aux_rank=64, aux_kernel=4, quant=None, wait=0,
                 dropout=0.0, gradient_checkpointing=False, device=None):
        super().__init__()
        if neurons < 2:
            raise ValueError(f"neurons must be >= 2, got {neurons}")
        if not isinstance(codebook, Codebook):
            raise TypeError(f"codebook must be a Codebook, got {type(codebook).__name__}")
        if codebook.width != neurons:
            raise ValueError(
                f"codebook.width ({codebook.width}) must equal neurons ({neurons}): "
                "codes are read in the state's own space")
        if depth < 0:
            raise ValueError(f"depth must be >= 0, got {depth}")
        if wait < 0:
            raise ValueError(f"wait must be >= 0, got {wait}")

        self.n = neurons
        self.depth = depth
        self.wait = wait
        self.rank = rank if rank is not None else max(4, neurons // 8)
        self.codebook = codebook
        self.gradient_checkpointing = gradient_checkpointing

        acts = {'gelu': nn.GELU(), 'silu': nn.SiLU(), 'tanh': nn.Tanh(), 'relu': nn.ReLU()}
        if activation not in acts:
            raise ValueError(f"activation must be one of {sorted(acts)}, got {activation!r}")
        self.act = acts[activation]

        self.W = nn.Parameter(torch.empty(neurons, neurons, device=device))
        self.b = nn.Parameter(torch.zeros(neurons, device=device))
        self.norm = nn.RMSNorm(neurons, device=device)

        # Scaled so one application preserves the state's variance; the
        # zero diagonal is enforced after, and by a grad hook thereafter.
        with torch.no_grad():
            nn.init.normal_(self.W, std=neurons ** -0.5)
            self.W.fill_diagonal_(0.0)
        self.W.register_hook(lambda g: g.clone().fill_diagonal_(0.0))

        if self_gain:
            # A latch, not a weight: gated shut, so `W` carries memory across
            # steps until a neuron earns the right to hold its own state.
            self.self_gain = nn.Parameter(torch.zeros(neurons, device=device))
            self.self_logit = nn.Parameter(torch.full((neurons,), -4.0, device=device))
        else:
            self.self_gain = self.self_logit = None

        if depth > 0:
            # One tensor per direction rather than `depth` of them: the branches
            # are independent, so they concatenate into a single matmul pair and
            # `depth` stops costing launches.
            self.U = nn.Parameter(torch.empty(neurons, depth * self.rank, device=device))
            # Zero, so every branch is the identity at step zero and `depth`
            # changes nothing until training moves it.
            self.V = nn.Parameter(torch.zeros(depth * self.rank, neurons, device=device))
            # Shared: one norm bounds what every branch reads, and a per-branch
            # norm would be `depth` more launches for the same bound.
            self.r_norm = nn.RMSNorm(neurons, device=device)
            with torch.no_grad():
                nn.init.normal_(self.U, std=neurons ** -0.5)
        else:
            self.U = self.V = self.r_norm = None

        self.drop = nn.Dropout(dropout)
        self.quant = quant
        self._enc_tail = None
        self.register_buffer('_h', torch.zeros(1, neurons, device=device), persistent=False)

        # The three optional branches, all zero-initialized at their output so
        # switching one on changes nothing at step 0. They read the whole
        # trajectory rather than one step, so they live outside the recurrence
        # and cost one set of kernels per call instead of one per token.
        self.attn = TemporalAttention(
            neurons, heads=attn_heads, kv_heads=attn_kv_heads,
            head_dim=attn_head_dim, window=attn_window, device=device) if attn else None
        self.plastic = Plasticity(neurons, dim=plastic_dim, device=device) if plastic else None

        if aux:
            # Causal depthwise conv over injected codes: local n-gram context
            # before the recurrence sees the token. Zero-init, so inert at step 0.
            self.enc = nn.Conv1d(neurons, neurons, kernel_size=aux_kernel,
                                 groups=neurons, bias=False, device=device)
            nn.init.zeros_(self.enc.weight)
            self.aux_kernel = aux_kernel
            # The readout adapter. Without it `h` must be the recurrent state
            # and point along a frozen random code at the same time; a rank-r
            # rotation separates those two jobs for 2*N*r parameters.
            self.dec_norm = nn.RMSNorm(neurons, device=device)
            self.dec_u = nn.Parameter(torch.empty(neurons, aux_rank, device=device))
            self.dec_v = nn.Parameter(torch.zeros(aux_rank, neurons, device=device))
            with torch.no_grad():
                nn.init.normal_(self.dec_u, std=neurons ** -0.5)
        else:
            self.enc = self.dec_norm = None
            self.dec_u = self.dec_v = None

    # ------------------------------------------------------------------ #

    def reset_caches(self):
        """Drop every cache that carries sequence history across calls."""
        if self.attn is not None:
            self.attn.reset_cache()
        if self.plastic is not None:
            self.plastic.reset_cache()
        self._enc_tail = None

    def _wait_steps(self, h, wcode, n, W, U, V, self_mult):
        """`n` extra applications of the core between two tokens.

        Depth here is time, so an extra step is literally one more application
        of `W` -- the same operator, on a state carrying the same context, with
        a marker on the input saying no token arrived. A transformer buying the
        same deliberation has to push a pause token through every layer; this
        architecture already is the loop, so the only cost is launches.

        The marker matters: with a zero input the step is `h @ W + b`, which is
        the same map the core applies when a token *is* absent at a padded
        position. A distinct frozen code lets the core tell "still thinking"
        from "nothing here", and it arrives through `in_gain` at a token's
        scale so the state stays in the regime the recurrence was trained on.
        """
        for _ in range(n):
            h = self._step(h, wcode, W, U, V, self_mult)
        return h

    def _encode(self, codes):
        """Add local n-gram context to the injected codes, causally.

        The conv's left context has to come from the previous call, not from
        zeros, or a token decoded one at a time sees `kernel-1` zeros where
        training gave it real predecessors. Carrying the tail is what makes
        chunked and whole-sequence runs agree -- and generation samples one
        token at a time, so without it the model generates from a distribution
        it was never scored on.
        """
        pad = self.aux_kernel - 1
        if self._enc_tail is None or self._enc_tail.shape[0] != codes.shape[0]:
            left = codes.new_zeros(codes.shape[0], pad, codes.shape[2])
        else:
            left = self._enc_tail
        padded = torch.cat([left, codes], dim=1)
        self._enc_tail = padded[:, -pad:].detach() if pad else None
        return codes + self.enc(padded.transpose(1, 2)).transpose(1, 2)

    def _step(self, h, code, W=None, U=None, V=None, self_mult=None):
        """One token: mix, inject, refine, normalize.

        The per-step invariants (`sigmoid(self_logit)`, the quantized operators)
        are passed in rather than recomputed: inside a `T`-step loop they would
        otherwise cost `T` times as many launches, and the loop is launch-bound.
        """
        W = self.W if W is None else W
        s = torch.addmm(self.b, h, W)
        if code is not None:
            s = s + code.to(s.dtype)
        if self_mult is not None:
            s = s + h * self_mult

        out = self.act(s)
        U = self.U if U is None else U
        if U is not None:
            # All branches in one matmul pair: `depth` is two launches, not
            # `3 * depth`, which is what makes it affordable here.
            V = self.V if V is None else V
            out = out + self.act(self.r_norm(out) @ U) @ V
        return self.norm(self.drop(out))

    def forward(self, ids, state=None, return_states=True, window=0, wait=None):
        """Run the core over token ids `(B, T)`.

        Returns `(logits, final_state)` where logits are `(B, T, V)` -- the
        prediction made *after* reading each token, so position `t` predicts
        token `t+1` and the caller shifts targets accordingly.

        `state` carries a previous call's final state for truncated BPTT; it
        arrives detached from the caller, and what is returned is live.

        `window` detaches the state every `window` steps, bounding how far
        credit travels back. 0 means the whole sequence, which is only usable
        at short `T`: the per-step Jacobian sits above 1, so the gradient grows
        exponentially in the backward length (measured 289x per doubling past
        128 steps). Every position is still predicted and still supervised --
        only the gradient's reach is cut, not the forward pass's.

        `wait` overrides `self.wait` for this call: that many extra core steps
        per token, on the frozen wait code, with no readout. One token still
        produces exactly one logit row, so nothing downstream changes shape.
        """
        if ids.ndim != 2:
            raise ValueError(f"ids must be (B, T), got shape {tuple(ids.shape)}")
        if window < 0:
            raise ValueError(f"window must be >= 0, got {window}")
        b, t = ids.shape
        wait = self.wait if wait is None else wait
        if wait < 0:
            raise ValueError(f"wait must be >= 0, got {wait}")

        if state is None:
            # A fresh state means a fresh stream: the branch caches belong to
            # the sequence, not to the module, and a stale one would let a new
            # document attend to the previous one's history.
            self.reset_caches()

        h = self._h.expand(b, -1) if state is None else state
        if h.shape != (b, self.n):
            raise ValueError(f"state must be ({b}, {self.n}), got {tuple(h.shape)}")
        h = h.to(self.W.dtype)

        # The whole sequence's codes in one gather, so the per-step work is the
        # matmul and nothing else.
        codes = self.codebook.inject(ids)
        if self.enc is not None:
            codes = self._encode(codes)

        # Hoisted out of the loop: these are constant across the `t` steps, and
        # recomputing them per step measured a 31% throughput loss on a
        # launch-bound recurrence.
        W = quantize_ste(self.W, self.quant)
        U = quantize_ste(self.U, self.quant) if self.U is not None else None
        V = quantize_ste(self.V, self.quant) if self.V is not None else None
        self_mult = (self.self_gain * torch.sigmoid(self.self_logit)
                     if self.self_logit is not None else None)
        wcode = self.codebook.inject_wait() if wait else None

        states = []
        for i in range(t):
            if window and i and i % window == 0:
                # The boundary: the state's value crosses, its history does not.
                h = h.detach()
            code = codes[:, i]
            if self.gradient_checkpointing and self.training:
                h = checkpoint.checkpoint(self._step, h, code, W, U, V, self_mult,
                                          use_reentrant=False)
            else:
                h = self._step(h, code, W, U, V, self_mult)
            if wait:
                # Between reading this token and predicting the next one. Only
                # the post-wait state is appended, so the branches below -- and
                # the readout -- see one entry per token: the attention cache
                # and the plasticity trace keep their horizons in tokens rather
                # than having them divided by the wait count.
                if self.gradient_checkpointing and self.training:
                    h = checkpoint.checkpoint(self._wait_steps, h, wcode, wait,
                                              W, U, V, self_mult, use_reentrant=False)
                else:
                    h = self._wait_steps(h, wcode, wait, W, U, V, self_mult)
            states.append(h)

        h_seq = torch.stack(states, dim=1)
        # The branches read the trajectory, so they run once per call rather
        # than once per token -- which is why they are affordable at all.
        if self.attn is not None:
            h_seq = h_seq + self.attn(h_seq)
        if self.plastic is not None:
            h_seq = h_seq + self.plastic(h_seq, window=window)
        if self.dec_u is not None:
            h_seq = h_seq + F.gelu(self.dec_norm(h_seq) @ self.dec_u) @ self.dec_v

        if not return_states:
            return self.codebook.logits(h_seq[:, -1:]), h
        return self.codebook.logits(h_seq), h

    @torch.no_grad()
    def generate(self, prompt, length=128, temperature=0.8, top_k=40, state=None,
                 wait=None):
        """Sample `length` tokens after `prompt` `(B, T)`. Greedy at temperature 0.

        `wait` applies per generated token exactly as in training, so decode
        spends the same computation per token that the scored distribution was
        produced with. Nothing about the wait steps reaches the output: the
        count of returned tokens is `length` whatever `wait` is.
        """
        was_training = self.training
        self.eval()
        logits, h = self.forward(prompt, state=state, return_states=False, wait=wait)
        out = []
        nxt = prompt[:, -1:]
        for _ in range(length):
            step = logits[:, -1].float()
            if temperature <= 0:
                nxt = step.argmax(-1, keepdim=True)
            else:
                step = step / temperature
                if top_k:
                    kth = step.topk(min(top_k, step.shape[-1]), dim=-1).values[:, -1:]
                    step = step.masked_fill(step < kth, float('-inf'))
                nxt = torch.multinomial(step.softmax(-1), 1)
            out.append(nxt)
            logits, h = self.forward(nxt, state=h, return_states=False, wait=wait)
        if was_training:
            self.train()
        return torch.cat(out, dim=1)

    # ------------------------------------------------------------------ #

    def param_report(self):
        """Trainable parameters by role, with the zero diagonal discounted.

        The number a parameter-efficiency claim is made against, so it is
        computed rather than quoted: `W`'s diagonal is structurally zero and
        does not count while `W` is being trained.
        """
        roles = {'core': [], 'refine': [], 'interface': [], 'modulation': []}
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name == 'W':
                roles['core'].append(p)
            elif name in ('U', 'V'):
                roles['refine'].append(p)
            elif name.startswith('codebook.'):
                roles['interface'].append(p)
            else:
                roles['modulation'].append(p)
        out = {k: sum(p.numel() for p in v) for k, v in roles.items()}
        if self.W.requires_grad:
            out['core'] -= self.n
        out['total'] = sum(out.values())
        return out

    def extra_repr(self):
        r = self.param_report()
        return (f"neurons={self.n}, depth={self.depth}, rank={self.rank}, "
                f"wait={self.wait}, params={r['total']:,} "
                f"(core {r['core']:,} + refine {r['refine']:,} + "
                f"interface {r['interface']:,} + mod {r['modulation']:,})")
