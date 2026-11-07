# Aleph

A language model that is one matrix.

Forked from [OdyssNet](https://github.com/theomgdev/OdyssNet) and decoupled from
it: Aleph keeps the idea that depth can be temporal rather than spatial, and
changes nearly everything about how a token gets in and out.

The problem it exists to solve is an accounting one. A conventional tiny
language model at `V=4096`, `N=512` pays `2·V·N = 4,194,304` parameters for an
embedding table and an output head, against `N² = 262,144` for the layer that
does the work. Ninety-four percent of the "model" is a dictionary, so a
parameter-efficiency figure quoted against the total is measuring the
dictionary.

## How

The token interface is frozen. Each token gets a fixed sparse ternary code
drawn from a seeded generator, and both directions read the same table:

```
inject(t) = H[t] · in_gain                 # V × N, frozen
logits    = h @ Hᵀ · out_scale + bias      # the same table, transposed
```

Trainable cost is `V + N + 1` — **4,609 against 4,194,304, 910× less** — and
what it asks in exchange is that the core learn to emit points in code space.

Random codes work because they are near-orthogonal. Two of them have inner
product `0 ± N^(-1/2)`, so `V` enters only logarithmically and a wider
vocabulary is nearly free. Measured at `V=4096, N=512`: rms overlap 0.0442
against the predicted 0.0442, at every code density. The worst-case overlap is
the part density controls (0.50 at `nnz=8`, 0.22 at `nnz=128`), and the readout
recovers every token exactly with a margin of 0.74–0.84 whatever it is.

Everything else is one square matrix `W` with a zero diagonal, read one token
per step:

```
s = h @ W + inject(x_t) + b
h = RMSNorm( φ(s) + Σ_k φ(RMSNorm(φ(s)) @ U_k) @ V_k )
```

`W` is the model. The `U_k, V_k` pairs are `N × rank` low-rank branches that buy
capacity inside a step rather than by echoing — see below for why that
distinction decides the design.

## What the measurements decided

Every number here was taken on an RTX 3060 Ti, and three of them overturned a
choice I had made from reasoning.

**The loop is launch-bound, severely.** 128× the batch cost 1.15× the time —
1,042 tok/s at batch 1 against 115,851 at batch 128. The clock tracks kernel
launches, not arithmetic, which means batch is nearly free and arithmetic
inside an existing launch is nearly free too.

**So depth must be parallel, not chained.** A chained refinement stack
(`r ← r + φ(norm(r) @ Uₖ) @ Vₖ`) needs `3·depth` sequential launches per token
and measured 4.59× slower at `depth=4` for 2.0× the parameters — it bought
capacity at a worse rate than it cost time. Concatenating the branches into one
wide matmul pair makes `depth` cost two launches whatever its value, and then
time *saturates*: depth 4, 8 and 16 all cost ~151–158 ms while parameters go
531k → 793k → 1.32M. Past depth 4 capacity is genuinely free.

**`torch.compile` is a slowdown here: 0.88×** (274.90 ms eager against 310.68
ms compiled). Inductor cannot fuse across a data-dependent Python loop and the
per-step graph is too small to win on its own.

**Depth narrows the usable learning rate rather than destabilizing the model.**
On a 64-sequence overfit, `depth=4` reached 0.0192 nats at lr 1e-3 and diverged
at 3e-3, where `depth=0` tolerated both and reached only 0.0626. The forward
scale through the stack grows 1.04–1.08×, so this is a step-size effect and not
an explosion — which is why the trainer warms up and why a depth comparison run
at one fixed rate measures the schedule instead of the mechanism.

**The unigram prior is worth paying `V` parameters for.** The frozen interface
plus a logit bias seeded from corpus counts reaches 5.6239 nats against the
distribution's true entropy of 5.5992 — essentially optimal, 3.9 bits below
uniform, for 4,609 parameters.

## Install

```bash
pip install -e ".[dev]"      # torch >= 2.4, Python >= 3.10
```

## Use

```bash
python tools/build_corpus.py --tokenizer bpe --vocab 4096 --stories 400000
python tools/experiment.py --mode smoke                   # wiring, in seconds
python tools/experiment.py --mode train --steps 6000 --sample
python tools/experiment.py --mode ablate --ablate depth=0,4,8,16
python tools/experiment.py --mode eval --tag main --sample # re-score, no GPU time
```

```python
from aleph import Codebook, AlephCore, Trainer

cb = Codebook(vocab=4096, width=512)          # 4,609 trainable
model = AlephCore(neurons=512, codebook=cb, depth=8)
print(model.param_report())
# {'core': 261632, 'refine': 524288, 'interface': 4609, 'modulation': 2560, ...}
```

Colab: [`notebooks/aleph_colab.ipynb`](notebooks/aleph_colab.ipynb) runs the
whole pipeline end to end, checkpointing to Drive so a disconnect resumes.

## Invariants

Breaking one of these silently changes what the architecture is, so they are
tested rather than documented:

- **`W.diagonal() == 0`.** A neuron's self-connection lives in `self_gain`, one
  parameter per neuron instead of a row of `W`, gated shut at init so the core
  has to earn the latch. Enforced by a gradient hook and after every step.
- **`|h| = √N`, exactly.** Per-step RMSNorm pins the state to that sphere, so
  explosion and vanishing are both structurally impossible regardless of `W`'s
  spectral radius — which is why no spectral constraint, clamp or learnable
  decay appears anywhere. Measured 15.999998 at `N=256`.
- **Every optional mechanism starts inert.** `V` is zero-initialized and
  `self_gain` is zero, so `depth` and the latch change nothing until training
  moves them: the same seed gives the same `W` at any depth, and an ablation arm
  differs from its baseline in one variable. Three parameter families
  consequently take **no gradient at step 0** — `U`, `r_norm` and `self_logit`
  sit behind a zero factor. That is the design working, not a detached graph;
  they open on their own schedule (measured: `U` reaches `|g| = 0.62` by step 6).
- **The code table never moves.** It is a buffer, not a parameter.

## What may and may not be claimed

`param_report()` returns four roles rather than one total, because a
parameter-efficiency claim lives or dies on which of them a reader counts.

Fair: the *core* — the trained computation — is `N² − N` parameters, and the
token interface costs `V + N + 1` where a learned embedding and head would cost
`2·V·N`. Quote bits per byte beside the trainable total and the cache it was
measured on.

Not fair: calling the frozen table free in the information-theoretic sense. It
is `V × nnz` fixed values chosen by a seed rather than by training, and a reader
is entitled to count its support however they like. Say *trainable* parameters,
which is what the number is.

Also not fair: comparing bits per byte across tokenizers or across rebuilt
caches. The metric is tokenizer-independent but it is defined against the
held-out split it was computed on, so quote the corpus's train/val token counts
with any comparison — including between two of our own runs.

## Layout

```
aleph/core/codebook.py    frozen token codes; the V+N+1 interface
aleph/core/core.py        the square recurrent core
aleph/data/stream.py      tokenizers, token cache, B-cursor stream
aleph/train/trainer.py    warmup/cosine, truncated BPTT, telemetry
tools/build_corpus.py     TinyStories -> uint16 token cache
tools/experiment.py       train / smoke / ablate / eval
tools/make_notebook.py    generates the Colab notebook
tests/test_aleph.py       40 tests, CPU, ~4s
```

`Research/` is gitignored: reference material, not library.

## License

MIT. See [LICENSE](LICENSE).
