"""Adaptive halting: let each token decide how long to think.

Fixed `wait` spends the same extra steps on every token, which is the worst
allocation available -- `the` and `and` pay the same price as the word whose
referent has to be resolved from forty tokens back. Measured at 800 matched
steps it bought -0.162 nats for zero parameters and then lost to the same
seconds spent on more steps by 0.284. The mechanism was never the problem. The
allocation was, and allocation is what this module is.

Halting as a distribution, not a decision
-----------------------------------------
"Stop now" is discrete and has no gradient, so follow PonderNet (Banino,
Balaguer & Blundell, 2021) and make the halting step a random variable. At
candidate state `h_k` the network emits

    lambda_k = sigmoid(h_k . w / sqrt(N) + b)

read as "halt here, given you have not already", and the probability of halting
at exactly `k` is the survival up to `k` times that:

    p_k = lambda_k * prod_{i<k} (1 - lambda_i),    p_M = prod_{i<M} (1 - lambda_i)

The last candidate is forced, which is where a maximum comes from: the cap is
not a rule bolted on afterwards and needs no penalty to enforce it, it is the
normalization of the distribution. `p` sums to one for any `lambda`, so there
is no configuration in which the model runs away.

What exits is the expected state, `sum_k p_k h_k`, rescaled to the norm the
recurrence actually produces (`sum_k p_k |h_k|`, rather than a hardcoded
`sqrt(N)`, so the learned norm gains are respected). The gradient into `w` is
then `<dL/dh_exit, h_k>`: a candidate earns weight exactly to the extent that
its direction reduces the loss. No straight-through estimator, no policy
gradient, no reward signal -- the halting head is trained by the same backward
pass as the rest of the model, and the loss never mentions it.

Why average the state and not the loss
--------------------------------------
PonderNet averages the *loss* over the halting distribution, which needs one
readout per candidate. Here the readout is `V`-wide and the branches
(attention, plasticity, the aux adapter) are applied once per token to the
committed state, so a per-candidate readout would be `M` copies of the widest
tensor in the graph -- 268 MB each at `B=256, T=64, V=4096` -- and would have
to be computed before the branches it is missing. Averaging the state costs one
extra kernel per token and leaves the readout exactly where it was.

The exchange is a cleaner early exit. Truncating the loop once the survival
probability drops below `eps` discards candidates whose total weight is below
`eps`, so the expected state is unchanged to that tolerance, and the rescaling
at the end absorbs the missing mass by construction. The approximation is the
mechanism rather than a train/inference mismatch: a token that wants one step
takes one.

The prior is the only thing that charges for time
-------------------------------------------------
Nothing in a cross-entropy prefers a short ponder, and a mixture of states is
free capacity, so the head would spread weight everywhere and spend the maximum
on every token -- which is fixed `wait` again, arrived at by gradient descent.
The regularizer is PonderNet's: a KL toward a geometric distribution with
halting rate `prior`, truncated to the same support. It fixes the mean the model
is charged for leaving and lets it leave wherever a token pays for the trip.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Ponder(nn.Module):
    """A learned halting law over extra core steps, for `N + 1` parameters.

    Args:
        neurons (int): State width `N`.
        max_wait (int): Most extra steps a token may take. The distribution's
            last candidate is forced, so this is a hard bound rather than a
            target -- and 0 makes the module exactly inert.
        prior (float): Halting rate of the geometric the KL pulls toward, so
            `1/prior - 1` is roughly the mean extra steps it is priced at.
            Default 0.5, i.e. one extra step on average.
        eps (float): Early-exit threshold on the survival probability. Applies
            outside training, where the batch is a decode rather than a step.
        device: Torch device.
    """

    def __init__(self, neurons, max_wait=4, prior=0.5, eps=0.01, device=None):
        super().__init__()
        if max_wait < 0:
            raise ValueError(f"max_wait must be >= 0, got {max_wait}")
        if not 0.0 < prior < 1.0:
            raise ValueError(f"prior must lie in (0, 1), got {prior}")
        self.n, self.max_wait, self.prior, self.eps = neurons, max_wait, prior, eps

        # Zero weight and a fixed bias: `lambda` starts state-independent, so a
        # pondering model and a plain one at the same seed share a core -- the
        # module draws no RNG at all, which is what keeps `ponder_max` an
        # ablation with one variable. State-dependence is learned.
        self.w = nn.Parameter(torch.zeros(neurons, device=device))
        self.bias = nn.Parameter(
            torch.full((), math.log(prior / (1 - prior)), device=device))

        self.reset_stats()

    def score(self, h):
        """Halting logit for `h` `(B, N)`.

        Divided by `sqrt(N)` because `h` lives on that sphere, so an AdamW step
        on `w` moves the logit at the same O(lr) scale as `bias` rather than
        `sqrt(N)` times faster along the state's mean direction.
        """
        return h @ self.w / (self.n ** 0.5) + self.bias

    # ------------------------------------------------------------------ #

    def reset_stats(self):
        """Begin a fresh accounting period. Called once per `forward`."""
        self._kl_sum = None
        self._steps_sum = None
        self._tokens = 0

    def observe(self, log_p):
        """Record one token's halting distribution `(B, K)`.

        Accumulates the KL the loss is charged and the expected step count the
        log reports. Kept here rather than returned so `AlephCore.forward`
        keeps its `(logits, state)` contract.
        """
        k = log_p.shape[-1]
        p = log_p.exp()
        # Truncated geometric: the support is what the loop actually ran, so
        # an early exit is compared against a prior over the same candidates.
        idx = torch.arange(k, device=log_p.device, dtype=log_p.dtype)
        log_prior = math.log(self.prior) + idx * math.log1p(-self.prior)
        log_prior = log_prior - torch.logsumexp(log_prior, 0)

        # Marginal KL rather than per-token KL: per-token KL(p_i || prior) is
        # minimized when every token outputs the *same* blurry geometric
        # distribution, which actively penalizes the head for distinguishing
        # tokens. Pricing the batch marginal p_bar = E_b[p] instead fixes the
        # population's step budget while leaving each token free to commit to
        # whichever step lowers its loss.
        p_bar = p.mean(0).clamp(min=1e-12)
        kl = (p_bar * (p_bar.log() - log_prior)).sum()
        per_row = (p * idx).sum(-1)
        steps = per_row.mean()
        self._kl_sum = kl if self._kl_sum is None else self._kl_sum + kl
        self._steps_sum = steps if self._steps_sum is None else self._steps_sum + steps
        self._tokens += 1
        self._last_per_row = per_row.detach()

    @property
    def kl(self):
        """Mean KL to the prior over the period, as a differentiable scalar."""
        if self._kl_sum is None:
            return None
        return self._kl_sum / max(self._tokens, 1)

    @property
    def mean_steps(self):
        """Expected extra steps per token over the period, as a float."""
        if self._steps_sum is None:
            return 0.0
        return (self._steps_sum / max(self._tokens, 1)).item()

    def extra_repr(self):
        return (f"neurons={self.n}, max_wait={self.max_wait}, "
                f"prior={self.prior}, params={self.n + 1}")
