"""The token interface, as a constant rather than a parameter.

A conventional tiny language model spends almost everything on its vocabulary.
At `V=4096` and width `N=512` an embedding table and an output head cost
`2 x V x N = 4.2M` while the recurrent core that does the computing costs
`N^2 = 262k`: 94% of the "model" is a lookup table, and a parameter-efficiency
figure quoted against that total measures the table.

So the table is not trained. Each token is assigned a fixed sparse ternary
vector `H[t]` once, drawn from a seeded generator, and the two directions both
read it:

    inject(t) = H[t] * in_gain                 (V x N, frozen)
    logits    = h @ H^T * out_scale + bias     (the same matrix, transposed)

What this costs is `N` input gains, one output scale and `V` logit biases --
`V + N + 1` against the core's `N^2`. What it asks in exchange is that the core
learn to emit points in code space, which a full `N x N` matrix applied `T`
times can do.

Why a random code separates
---------------------------
Rows carry `k` nonzeros of magnitude `k^-1/2` at random positions with random
signs. Two rows collide on `k^2/N` positions on average and each collision
contributes `+-1/k`, so `<H[i], H[j]>` has mean 0 and standard deviation
`~N^-1/2` -- 0.044 at `N=512`, against a self-inner-product of 1. That is the
Johnson-Lindenstrauss regime, and `V` enters only logarithmically, so a wider
vocabulary is nearly free.

Sparsity does not change that typical inner product, but it does set the
worst-case one: a pair sharing `m` of its `k` positions reaches `m/k`, so the
tail over all `V^2/2` pairs is governed by `k` while the rms stays at `N^-1/2`.
The tail stops shrinking around `k=128` and the readout margin stops improving
around `k=64`, which is where `nnz` defaults. A frozen table is stored dense
either way, so density costs neither parameters nor time.

`code='ngram'` instead sums hashed character n-grams of the token's own string,
which makes the code carry morphology: "running" and "runner" share n-grams and
therefore land near each other before any training happens. It buys a prior and
gives up exact orthogonality, so it is a measurement rather than a default.

The logit bias is initialized from corpus unigram counts when they are
available. Token frequency is most of a language model's first nat, and paying
`V` parameters for it is cheaper than making the core rediscover it.
"""

import hashlib

import torch
import torch.nn as nn


def _sparse_ternary(vocab, width, nnz, seed, device):
    """`(V, N)` rows with `nnz` entries of magnitude `nnz^-1/2` and random signs."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    out = torch.zeros(vocab, width)
    # Per-row sampling without replacement: argsort of noise is the cheapest
    # way to get `nnz` distinct columns for every row in one shot.
    cols = torch.rand(vocab, width, generator=g).argsort(dim=1)[:, :nnz]
    signs = (torch.randint(0, 2, (vocab, nnz), generator=g).float() * 2.0 - 1.0)
    out.scatter_(1, cols, signs * (nnz ** -0.5))
    return out.to(device)


def _ngram_code(tokens, width, nnz, seed, device, orders=(2, 3, 4)):
    """Character n-gram hashing, so related spellings get related codes.

    Each n-gram hashes to one column and one sign; a token's code is the mean of
    its n-grams', renormalized. Tokens whose string is unavailable or too short
    fall back to a hash of the token id, which is just a one-token n-gram.
    """
    out = torch.zeros(len(tokens), width)
    for row, text in enumerate(tokens):
        s = f"<{text}>" if text else f"<#{row}>"
        grams = [s[i:i + n] for n in orders for i in range(max(0, len(s) - n + 1))]
        if not grams:
            grams = [s]
        for gram in grams:
            h = int.from_bytes(
                hashlib.blake2b(gram.encode('utf-8', 'replace'),
                                digest_size=8, key=seed.to_bytes(8, 'little')).digest(),
                'little')
            out[row, h % width] += 1.0 if (h >> 63) & 1 else -1.0
    # Unit rows, matching the sparse code's scale, so the injection gain and
    # the readout mean the same thing whichever construction produced the table.
    norm = out.norm(dim=1, keepdim=True).clamp(min=1e-6)
    return (out / norm).to(device)


class Codebook(nn.Module):
    """Frozen token codes plus the few trainable scalars that read them.

    Args:
        vocab (int): Vocabulary size.
        width (int): State width `N`; codes live in the state's own space.
        nnz (int): Nonzeros per code. Default 64. Sets the worst-case inner
            product between two codes (`~1/nnz`); the typical one is `N^-1/2`
            whatever the density. Free in parameters and in time.
        code (str): `'random'` (sparse ternary) or `'ngram'` (hashed character
            n-grams, needs `tokens`).
        tokens (list[str], optional): Token strings, for `code='ngram'`.
        in_gain (bool): Per-neuron gain on injection (`N` parameters). Default
            True -- the core can scale the whole code down but not reweight the
            directions it arrives on.
        logit_bias (bool): Per-token output bias (`V` parameters). Default True.
        seed (int): Code generator seed. Part of the architecture: a model only
            reads codes identical to the ones it trained against.
    """

    def __init__(self, vocab, width, nnz=64, code='random', tokens=None,
                 in_gain=True, logit_bias=True, seed=0xA1EF, device=None):
        super().__init__()
        if vocab < 2:
            raise ValueError(f"vocab must be >= 2, got {vocab}")
        if width < 2:
            raise ValueError(f"width must be >= 2, got {width}")
        if not 1 <= nnz <= width:
            raise ValueError(f"nnz must lie in [1, {width}], got {nnz}")
        if code not in ('random', 'ngram'):
            raise ValueError(f"code must be 'random' or 'ngram', got {code!r}")
        if code == 'ngram' and tokens is None:
            raise ValueError("code='ngram' needs `tokens` (the token strings)")
        if code == 'ngram' and len(tokens) != vocab:
            raise ValueError(f"tokens has {len(tokens)} entries, vocab is {vocab}")

        self.vocab, self.width, self.nnz, self.code, self.seed = vocab, width, nnz, code, seed

        table = (_sparse_ternary(vocab, width, nnz, seed, device) if code == 'random'
                 else _ngram_code(tokens, width, nnz, seed, device))
        # Persistent: regenerating from the seed would agree, but a checkpoint
        # that carries its own interface cannot be read against a different one
        # by accident.
        self.register_buffer('H', table)

        # The wait code: one more row, from a disjoint seed, kept out of `H` so
        # it is not in the readout's span at all. A token the softmax cannot
        # emit cannot be suppressed by the softmax either -- at every supervised
        # position the gradient of -log p(target) pushes every non-target logit
        # down, so a wait token inside the vocabulary would be trained toward
        # never being produced. It is injected, never predicted.
        self.register_buffer('wait_code',
                             _sparse_ternary(1, width, nnz, seed ^ 0x5A17, device)[0])

        self.in_gain = nn.Parameter(torch.ones(width, device=device)) if in_gain else None
        self.out_scale = nn.Parameter(torch.ones((), device=device))
        self.bias = nn.Parameter(torch.zeros(vocab, device=device)) if logit_bias else None

    def set_unigram(self, counts):
        """Seed the logit bias with log token frequency.

        Unigram statistics are a large and entirely predictable share of the
        loss; paying `V` parameters for them beats spending core capacity.
        Counts are smoothed, normalized and centered -- an unseen token gets the
        floor rather than `-inf`, and the mean sits at zero so the core still
        starts from a uniform prediction.
        """
        if self.bias is None:
            return
        with torch.no_grad():
            c = torch.as_tensor(counts, dtype=torch.float32, device=self.bias.device)
            if c.shape != self.bias.shape:
                raise ValueError(f"counts has shape {tuple(c.shape)}, need {tuple(self.bias.shape)}")
            logp = (c + 0.5).log()
            self.bias.copy_(logp - logp.mean())

    def inject(self, ids):
        """Codes for token ids `(B,)` or `(B, T)`, shaped `(..., N)`.

        `-1` is the "inject nothing" sentinel and returns a zero row, so a
        padded position perturbs the state exactly as much as an absent one.
        """
        H = self.H
        safe = ids.clamp(min=0)
        vec = H[safe]
        if self.in_gain is not None:
            vec = vec * self.in_gain
        return vec * (ids >= 0).unsqueeze(-1).to(vec.dtype)

    def inject_wait(self):
        """The wait code, read through the same gain as any injected token.

        Shaped `(N,)` and broadcast by the caller. Read through `in_gain` so a
        wait step arrives on the same directions, at the same scale, as a token
        -- the core gets one more application of `W` with a marker on the input,
        not a differently-scaled perturbation it has to learn to tolerate.
        """
        vec = self.wait_code
        if self.in_gain is not None:
            vec = vec * self.in_gain
        return vec

    def logits(self, h):
        """Scores over the vocabulary for state `h`, shaped `(..., N) -> (..., V)`."""
        out = (h @ self.H.t().to(h.dtype)) * self.out_scale.to(h.dtype)
        if self.bias is not None:
            out = out + self.bias.to(h.dtype)
        return out

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def coherence(self):
        """Largest off-diagonal code inner product -- how close two tokens get.

        The quantity that decides whether the readout can separate the
        vocabulary at all; `~N^-1/2` for a random code. Materializes a `V x V`
        matrix, so it is a diagnostic rather than something to call in a loop.
        """
        with torch.no_grad():
            H = self.H.float()
            G = (H @ H.t()).abs()
            G.fill_diagonal_(0.0)
            return G.max().item()

    def extra_repr(self):
        return (f"vocab={self.vocab}, width={self.width}, nnz={self.nnz}, "
                f"code={self.code}, trainable={self.num_params()}")
