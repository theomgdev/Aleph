"""Fast weights: a per-stream memory matrix that is written, not trained.

The core's `(B, N)` state is a vector, so everything it remembers competes for
the same `N` coordinates. A fast-weight matrix gives each stream a second,
larger memory -- `(B, D, D)` -- that no parameter pays for: it is an activation,
written at test time by the sequence itself.

OdyssNet wrote it by plain Hebbian accumulation, `S += v k^T`. The problem is
capacity: language keys are not orthogonal, so reading back `S q` returns the
wanted value plus `sum_j <k_j, q> v_j` over every other write, and that
crosstalk grows with the sequence. Decay bounds the horizon without curing the
interference.

The delta rule writes the *error* instead of the value:

    err_t = v_t - S_{t-1} k_t
    S_t   = decay * S_{t-1} + (err_t k_t^T) * beta

If `k_t` already retrieves `v_t`, the write is zero. If it retrieves something
stale, the write erases exactly that and replaces it along `k_t`. This is one
step of online least squares, and it is what DeltaNet, Titans and TTT all turn
out to be doing (see `Research/24-fast-weights-and-plasticity.md`).

Two things make this *Aleph's* version rather than a port.

**Temporal and spatial keys in one trace.** The key is
`k = norm(k_temp(h_{t-1}) + k_spat(h_t))`: the temporal half makes the write
directed, `h_{t-1} -> h_t`, which is the association a sequence model needs; the
spatial half makes it content-addressed by the current state. Both paths share
the trace, so `'both'` costs one set of kernels rather than two.

**Synaptic resolution.** `beta` is a full `(D, D)` matrix of per-synapse
plasticity gates, not a scalar or a per-row vector. Which synapses are plastic
is learned; how much they write is the same question at every synapse. It costs
`D^2` parameters and is the resolution OdyssNet's `hebb_res` never reached.

The trace crosses calls detached, so each stream carries its own associative
memory for the length of a document while gradient stays inside the window.
`o_proj` is zero-initialized: the branch contributes nothing until trained.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Plasticity(nn.Module):
    """Delta-rule fast weights at synaptic resolution, carried across calls.

    Args:
        width (int): State width `N`.
        dim (int): Fast-weight width `D`. The matrix is `D x D` per stream, so
            this is the memory knob. Default 32, which measured as good as 64
            at a quarter of the parameters.
        device: Torch device.
    """

    def __init__(self, width, dim=32, device=None):
        super().__init__()
        self.width, self.dim = width, dim

        self.norm = nn.RMSNorm(width, device=device)
        self.k_temp = nn.Linear(width, dim, bias=False, device=device)
        self.k_spat = nn.Linear(width, dim, bias=False, device=device)
        self.v_proj = nn.Linear(width, dim, bias=False, device=device)
        self.q_proj = nn.Linear(width, dim, bias=False, device=device)
        self.o_proj = nn.Linear(dim, width, bias=False, device=device)
        self.read_norm = nn.RMSNorm(dim, device=device)

        # Raw logits through sigmoid: their zero is not a neutral point, which
        # is why the optimizer must never decay them.
        self.beta = nn.Parameter(torch.full((dim, dim), -2.5, device=device))
        self.decay = nn.Parameter(torch.full((dim,), 2.5, device=device))

        with torch.no_grad():
            for proj in (self.k_temp, self.k_spat, self.v_proj, self.q_proj):
                nn.init.normal_(proj.weight, std=width ** -0.5)
            nn.init.zeros_(self.o_proj.weight)
        self.reset_cache()

    def reset_cache(self):
        self._S = None
        self._last = None

    def forward(self, h_seq, window=32):
        """`(B, T, N) -> (B, T, N)`, the contribution to add to the trajectory."""
        b, t, _ = h_seq.shape
        if self._S is not None and self._S.shape[0] != b:
            self.reset_cache()

        x = self.norm(h_seq)
        # The previous state for position 0 comes from the last call, so the
        # temporal pairing survives a chunk boundary like everything else here.
        prev = (torch.zeros(b, 1, self.width, device=x.device, dtype=x.dtype)
                if self._last is None else self._last)
        x_prev = torch.cat([prev, x[:, :-1]], dim=1)
        self._last = x[:, -1:].detach()

        k = F.normalize(self.k_temp(x_prev) + self.k_spat(x), dim=-1)
        v = self.v_proj(x)
        q = F.normalize(self.q_proj(x), dim=-1)

        beta = torch.sigmoid(self.beta).unsqueeze(0)
        decay = torch.sigmoid(self.decay).view(1, self.dim, 1)

        S = (torch.zeros(b, self.dim, self.dim, device=x.device, dtype=x.dtype)
             if self._S is None else self._S)
        reads = []
        for i in range(t):
            if window and i and i % window == 0:
                S = S.detach()
            ki, vi, qi = k[:, i, :, None], v[:, i, :, None], q[:, i, :, None]
            # Read before write, so a position never retrieves its own answer.
            reads.append((S @ qi).squeeze(-1))
            err = vi - (S @ ki)
            S = decay * S + (err @ ki.transpose(-1, -2)) * beta

        self._S = S.detach()
        return self.o_proj(self.read_norm(torch.stack(reads, dim=1)))

    def extra_repr(self):
        return f"width={self.width}, dim={self.dim}, state_per_stream={self.dim ** 2}"
