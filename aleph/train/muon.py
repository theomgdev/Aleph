"""Muon for the core, AdamW for everything else.

Muon takes the steepest-descent step under the spectral norm: it orthogonalizes
the momentum matrix, so the update's singular values are all 1 and the step
moves every direction of `W` equally. Jordan et al. (2024) measure 1.3-1.4x
faster convergence than tuned AdamW on models up to ~520M.

Why it is interesting *here* specifically, beyond the speedup. Aleph's gradient
explodes because the recurrence's per-step Jacobian sits slightly above 1
(measured 1.0647 at N=512), and that number is governed by `W`'s spectral
radius. AdamW rescales each entry independently and so has no view of the
spectrum at all; Muon's update is orthogonal by construction, which keeps the
singular values from drifting apart. If it holds the Jacobian at or below 1, the
backward window can be longer for the same gradient norm -- and the window is
the one thing currently limiting how far credit travels.

The orthogonalization is a 5-step Newton-Schulz iteration rather than an SVD:

    X_0 = M / (||M||_F + eps)
    X_{k+1} = a X_k + b (X_k X_k^T) X_k + c (X_k X_k^T)^2 X_k

with quintic coefficients tuned so the iteration converges in bf16. It costs
`~6 N^3` FLOPs, which on a launch-bound model is five more kernels per step on
a matrix that is already resident -- cheap next to the recurrence.

It *approximately* orthogonalizes, and how closely depends on the input's
conditioning rather than on precision. Measured on a random matrix with
condition number 2,835, the smallest singular value goes 0.0053 -> 0.0614 at 5
steps and plateaus at ~0.68 by 8, while the largest stays near 1.15; bf16 and
fp32 agree to within noise, so the cheap dtype costs nothing. Five steps is the
published recipe and is enough for a momentum matrix, which is far better
conditioned than that worst case -- but the update is not an isometry, and a
claim that rests on its singular values being exactly 1 would be wrong.

Two scope rules are not negotiable, and both come from the derivation rather
than from practice. Muon is steepest descent under an operator norm, so it
applies only to 2D matrices that act as operators: `W`, `U` and `V` qualify.
Embeddings, readout heads and anything 1D do not -- token frequencies are
Zipfian and orthogonalizing across them destroys rare-token gradients, so they
stay on AdamW, where coordinate-wise scaling is the right geometry. Aleph's
interface is frozen, which removes most of that risk by construction, but the
logit bias and every norm gain still belong on AdamW.

One Aleph-specific correction: Newton-Schulz returns a dense matrix, so it
fills the zero diagonal `W` is required to keep. The diagonal is re-zeroed after
the update, the same way the gradient hook handles the backward pass.
"""

import torch


@torch.no_grad()
def orthogonalize(M, steps=5, eps=1e-7):
    """Approximately the nearest semi-orthogonal matrix to `M`.

    Runs in bf16: measured, it agrees with fp32 to within noise, and this is
    the one place in the step where precision buys nothing. `steps=5` is the
    published recipe; the residual non-orthogonality it leaves is set by `M`'s
    conditioning and plateaus by ~8 steps (see the module docstring).
    """
    if M.ndim != 2:
        raise ValueError(f"orthogonalize expects a 2D matrix, got {M.ndim}D")
    a, b, c = 3.4445, -4.7750, 2.0315
    X = M.bfloat16()
    X = X / (X.norm() + eps)
    # The iteration converges on the smaller dimension, so operate on the
    # transpose when the matrix is wide and transpose back at the end.
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * (A @ A)) @ X
    if transposed:
        X = X.T
    return X.to(M.dtype)


class MuonAdamW(torch.optim.Optimizer):
    """Muon on 2D operator matrices, AdamW on everything else.

    One optimizer rather than two, because the two halves share a step counter
    and a schedule and keeping them in separate objects means keeping those in
    sync by hand.

    Args:
        muon_params: 2D matrices that act as operators (`W`, `U`, `V`).
        adamw_params: everything else -- biases, norm gains, the codebook's
            scalars and logit bias.
        lr (float): Peak rate for the AdamW half.
        muon_lr (float, optional): Peak rate for the Muon half. Defaults to
            `lr`; the orthogonal update has unit singular values, so its
            natural scale differs from AdamW's and this is the knob that says
            by how much.
        momentum (float): Muon's momentum. Default 0.95.
        nesterov (bool): Nesterov momentum for the Muon half. Default True.
        betas, eps: AdamW's.
        weight_decay (float): Decoupled, applied to the Muon half only --
            the AdamW half here is all modulation, whose zero is not neutral.
        zero_diag (bool): Re-zero the diagonal of any square Muon matrix after
            the update. Default True: Newton-Schulz returns a dense matrix and
            `W`'s diagonal is structurally zero.
    """

    def __init__(self, muon_params, adamw_params, lr=1e-3, muon_lr=None,
                 momentum=0.95, nesterov=True, betas=(0.9, 0.95), eps=1e-8,
                 weight_decay=0.01, zero_diag=True):
        muon_params, adamw_params = list(muon_params), list(adamw_params)
        bad = [tuple(p.shape) for p in muon_params if p.ndim != 2]
        if bad:
            raise ValueError(
                f"Muon applies only to 2D operator matrices; got {bad}. "
                "1D parameters have trivial singular values and belong on AdamW.")

        groups = []
        if muon_params:
            groups.append({'params': muon_params, 'use_muon': True,
                           'lr': muon_lr if muon_lr is not None else lr,
                           'momentum': momentum, 'nesterov': nesterov,
                           'weight_decay': weight_decay, 'zero_diag': zero_diag})
        if adamw_params:
            groups.append({'params': adamw_params, 'use_muon': False, 'lr': lr,
                           'betas': betas, 'eps': eps, 'weight_decay': 0.0,
                           'zero_diag': False})
        if not groups:
            raise ValueError("no parameters given")
        super().__init__(groups, {})

    def set_lr(self, lr, muon_scale=1.0):
        """Set both halves from one schedule value.

        `muon_scale` is how much more step the Muon half takes, and it must be
        passed on every call: this overwrites `lr`, so a `muon_lr` given at
        construction does not survive the first schedule update. A probe that
        set `muon_lr` and then called `set_lr(lr)` per step silently ran the
        default ratio and produced an arm identical to its neighbour.
        """
        for g in self.param_groups:
            g['lr'] = lr * (muon_scale if g['use_muon'] else 1.0)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            if group['use_muon']:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(p)
                    buf = state['momentum_buffer']
                    buf.mul_(group['momentum']).add_(p.grad)
                    g = (p.grad.add(buf, alpha=group['momentum'])
                         if group['nesterov'] else buf)

                    update = orthogonalize(g)
                    # Moonshot AI RMS-matching rule (Liu et al. 2025,
                    # arXiv:2502.16982): an orthogonal A x B update has entry
                    # RMS 1 / sqrt(max(A, B)), whereas AdamW's update RMS sits
                    # near 0.2. Scaling by 0.2 * sqrt(max(A, B)) puts Muon on
                    # AdamW's step scale across every width and aspect ratio.
                    scale = 0.2 * (max(p.shape[0], p.shape[1]) ** 0.5)
                    if group['weight_decay']:
                        p.mul_(1 - group['lr'] * group['weight_decay'])
                    p.add_(update, alpha=-group['lr'] * scale)
                    if group['zero_diag'] and p.shape[0] == p.shape[1]:
                        p.fill_diagonal_(0.0)
            else:
                beta1, beta2 = group['betas']
                for p in group['params']:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if 'step' not in state:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)
                    state['step'] += 1
                    k = state['step']
                    state['exp_avg'].mul_(beta1).add_(p.grad, alpha=1 - beta1)
                    state['exp_avg_sq'].mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
                    denom = (state['exp_avg_sq'] / (1 - beta2 ** k)).sqrt_().add_(group['eps'])
                    p.addcdiv_(state['exp_avg'] / (1 - beta1 ** k), denom, value=-group['lr'])
        return loss

    @classmethod
    def from_model(cls, model, **kw):
        """Split parameters by shape, with one exception.

        Every 2D parameter here is an operator -- it maps one vector space to
        another -- and orthogonalizing its update is what Muon is for. Gains,
        biases and the state vector are 1D and belong to AdamW.

        The exception is `Plasticity.beta`: it is 2D but it is not an operator,
        it is a `D x D` field of independent sigmoid logits. Orthogonalizing it
        would mix gates that have nothing to do with each other.
        """
        muon, adamw = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            is_operator = p.ndim == 2 and not name.endswith('beta')
            (muon if is_operator else adamw).append(p)
        return cls(muon, adamw, **kw)
