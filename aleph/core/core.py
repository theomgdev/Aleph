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
import torch.utils.checkpoint as checkpoint

from .codebook import Codebook


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
        dropout (float): Applied to the state after the activation. Default 0.
        gradient_checkpointing (bool): Recompute steps in the backward pass.
            Trades ~30% time for a step-count-independent memory profile.
        device: Torch device.
    """

    def __init__(self, neurons, codebook, depth=4, rank=None, activation='gelu',
                 self_gain=True, dropout=0.0, gradient_checkpointing=False,
                 device=None):
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

        self.n = neurons
        self.depth = depth
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
        self.register_buffer('_h', torch.zeros(1, neurons, device=device), persistent=False)

    # ------------------------------------------------------------------ #

    def _step(self, h, code):
        """One token: mix, inject, refine, normalize. `code` may be None."""
        s = h @ self.W + self.b
        if code is not None:
            s = s + code.to(s.dtype)
        if self.self_logit is not None:
            s = s + h * self.self_gain * torch.sigmoid(self.self_logit)

        out = self.act(s)
        if self.U is not None:
            # All branches in one matmul pair: `depth` is two launches, not
            # `3 * depth`, which is what makes it affordable here.
            out = out + self.act(self.r_norm(out) @ self.U) @ self.V
        return self.norm(self.drop(out))

    def forward(self, ids, state=None, return_states=True, window=0):
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
        """
        if ids.ndim != 2:
            raise ValueError(f"ids must be (B, T), got shape {tuple(ids.shape)}")
        if window < 0:
            raise ValueError(f"window must be >= 0, got {window}")
        b, t = ids.shape

        h = self._h.expand(b, -1) if state is None else state
        if h.shape != (b, self.n):
            raise ValueError(f"state must be ({b}, {self.n}), got {tuple(h.shape)}")
        h = h.to(self.W.dtype)

        # The whole sequence's codes in one gather, so the per-step work is the
        # matmul and nothing else.
        codes = self.codebook.inject(ids)

        states = [] if return_states else None
        for i in range(t):
            if window and i and i % window == 0:
                # The boundary: the state's value crosses, its history does not.
                h = h.detach()
            code = codes[:, i]
            if self.gradient_checkpointing and self.training:
                h = checkpoint.checkpoint(self._step, h, code, use_reentrant=False)
            else:
                h = self._step(h, code)
            if return_states:
                states.append(h)

        if not return_states:
            return self.codebook.logits(h).unsqueeze(1), h
        return self.codebook.logits(torch.stack(states, dim=1)), h

    @torch.no_grad()
    def generate(self, prompt, length=128, temperature=0.8, top_k=40, state=None):
        """Sample `length` tokens after `prompt` `(B, T)`. Greedy at temperature 0."""
        was_training = self.training
        self.eval()
        logits, h = self.forward(prompt, state=state, return_states=False)
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
            logits, h = self.forward(nxt, state=h, return_states=False)
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
                f"params={r['total']:,} "
                f"(core {r['core']:,} + refine {r['refine']:,} + "
                f"interface {r['interface']:,} + mod {r['modulation']:,})")
