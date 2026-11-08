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

Three branches then read the whole trajectory, once per call rather than once
per token, which is what makes them affordable:

```
h_seq += attention(h_seq)        # queries a cache of earlier states, over time
h_seq += plasticity(h_seq)       # a D×D fast-weight matrix written per stream
h_seq += readout_adapter(h_seq)  # rotates the state into the codebook's frame
```

**Attention** has no layers to sit between, so it attends along time: one KV
head, RoPE at absolute stream position, and the cache crosses forward calls
detached — so a short chunk still sees a long past.

**Plasticity** is a fast-weight matrix per stream, written by the sequence
rather than trained. It writes the *error*, `S ← decay·S + (v − Sk)kᵀ·β`, so a
repeat write is a no-op and a stale association is erased rather than
accumulated; `β` is a full per-synapse gate field. The matrix is an activation,
not a parameter: 1,024 floats of memory per stream at `D=32`, against the 512
the state vector holds.

**The readout adapter** is the cheapest and the most effective per parameter.
Without it the state has to be recurrent memory *and* point along a frozen
random code at once; a rank-64 rotation separates those two jobs.

All three are zero-initialized at their output, so switching one on changes
nothing at step 0 and an ablation differs in exactly one variable.

### Optionally, more than one step per token

Depth is time here, so a token can be given more computation by applying `W`
again with no token arriving — marked by a frozen code held outside `H`, so it
is injected and never predicted. That placement is the mechanism, not a
detail: a code the readout can emit is one the softmax is trained to suppress,
because every supervised position pushes every non-target logit down. Kept out
of the table, it needs no loss term and no reward to survive.

```
wait=k        k extra steps on every token
ponder_max=M  up to M, allocated per token by a learned halting law (N+1 params)
```

The adaptive form follows PonderNet: `λ_k = σ(h_k·w/√N + b)` reads as "stop
here, given you have not already", the chance of stopping at exactly `k` is
`λ_k ∏_{i<k}(1−λ_i)`, and the last candidate absorbs the surviving mass. That
sums to one for any weights, so the ceiling is the support of a distribution
rather than a rule — it cannot be exceeded and needs no penalty to hold. The
state leaves as the norm-rescaled expectation `Σ_k p_k h_k`, which is what makes
the head trainable by the same backward pass as `W`, with no straight-through
estimator anywhere. A KL toward a truncated geometric prices the budget.

Both default to 0 and are bit-identical there. `ponder_max` is the better model
per token (−0.301 nats at 800 matched steps) and the worse use of a second,
because a mixture over candidates has to compute all of them — the honest
accounting is in [`MEASUREMENTS.md`](MEASUREMENTS.md).

## What the measurements decided

Every number here was taken on an RTX 3060 Ti, and several of them overturned a
choice I had made from reasoning.

**The loop is launch-bound, severely.** 128× the batch cost 1.15× the time —
1,042 tok/s at batch 1 against 115,851 at batch 128. The clock tracks kernel
launches, not arithmetic, which means batch is nearly free and arithmetic
inside an existing launch is nearly free too. At `B=64, T=256` the recurrence
is 308.3 ms of a 314.3 ms step, while the readout and cross-entropy over
`V=4096` — 206 GFLOPs against the recurrence's 25.7 — take 19.6 ms. Ninety-eight
percent of the time on eleven percent of the arithmetic.

**So the sequence should be short and the batch wide.** The state and both
branch caches cross forward calls, which makes `seq` a launch granularity
rather than a context length: a token sees unbroken history and credit reaches
`window` tokens back either way. At a fixed 16,384 tokens per step and
identical peak VRAM (1,660 MiB), `(256, 64)` ran 3.6× faster than `(64, 256)` —
158,085 against 50,439 tok/s — and scored *lower*, because a step averages over
256 independent stream positions instead of 64. Largest speedup in the project,
no cost, so it is the default.

**So depth must be parallel, not chained.** A chained refinement stack
(`r ← r + φ(norm(r) @ Uₖ) @ Vₖ`) needs `3·depth` sequential launches per token
and measured 4.59× slower at `depth=4` for 2.0× the parameters — it bought
capacity at a worse rate than it cost time. Concatenating the branches into one
wide matmul pair makes `depth` cost two launches whatever its value, and then
time *saturates*: depth 4, 8 and 16 all cost ~151–158 ms while parameters go
531k → 793k → 1.32M. Past depth 4 capacity is genuinely free.

**And autocast is a slowdown, not a speedup: 0.76×.** bf16 autocast inside the
step loop casts the same three matrices once per token — 768 extra kernels per
forward pass at `T=256` — on a loop that is already latency-bound. TF32 runs
31% faster *and* more precisely. `torch.compile` loses too (0.88×): Inductor
cannot fuse across a data-dependent Python loop.

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

**Muon beats AdamW by 0.531 nats at zero parameter cost**, which makes it the
largest single win in the table and the default. The advantage is in
convergence rate rather than the final optimum, so it is invisible early: at 25
steps AdamW is still ahead. A probe short enough to be cheap is long enough to
reach the wrong conclusion.

**int8 weights are free; ternary is not.** QAT on the recurrent operators costs
+0.014 nats at int8 (inside run-to-run noise, 4× smaller storage), +0.267 at
int4 and +0.477 at ternary. That last figure independently reproduces the
published +0.50 for sub-200M models, so it is recorded as a negative result:
this architecture at this scale does not support 1.58-bit weights.

**Extra steps per token buy loss, and lose the clock.** Applying `W` more than
once per token is worth −0.162 nats fixed and −0.301 nats with a learned
allocation, at matched optimizer steps and almost no parameters. Given the same
*seconds* instead, plain training wins by 0.284–0.682 nats, and the gap widens
with budget. The learned head is genuinely state-dependent — it beats a
constant at the same mean cost — but what it learned is a depth *ensemble*
rather than a stopping rule: truncating the low-probability tail at inference
costs +0.43 nats, so every candidate step has to run. Both axes ship inert.

Full tables, including the arms that lost, are in
[`MEASUREMENTS.md`](MEASUREMENTS.md).

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
python tools/experiment.py --mode ablate \
       --ablate branches=none+adamw,none,aux,attn,plastic,attn+plastic+aux
python tools/experiment.py --mode ablate --ablate quant=none,int8,int4,ternary
python tools/experiment.py --mode ablate --ablate wait=0,1,2,4
python tools/experiment.py --mode ablate --ablate ponder_max=0,2,4,8
python tools/experiment.py --mode depends --tag main         # what trained weights rely on
python tools/experiment.py --mode eval --tag main --sample # re-score, no GPU time
```

```python
from aleph import Codebook, AlephCore, Trainer

cb = Codebook(vocab=4096, width=512)          # 4,609 trainable
model = AlephCore(neurons=512, codebook=cb, depth=8)
print(model.param_report())
# {'core': 261632, 'refine': 524288, 'interface': 4609, 'modulation': 236640, ...}
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
- **Every optional mechanism starts inert.** `V` is zero-initialized, `self_gain`
  is zero, and all three branches are zero-initialized at their output
  projection, so `depth`, the latch, attention, plasticity and the adapter
  change nothing until training moves them: the same seed gives the same `W` in
  every configuration, and an ablation arm differs from its baseline in one
  variable. Several parameter families consequently take **no gradient at step
  0** — everything behind a zero factor. That is the design working, not a
  detached graph; they open on their own schedule (measured: `U` reaches
  `|g| = 0.62` by step 6, and all 31 tensors have moved by step 12).
- **The caches belong to the sequence, not the module.** Attention's KV cache
  and plasticity's fast-weight trace cross forward calls carrying value but not
  gradient — the same contract the hidden state has — and are reset whenever
  `state=None` arrives, so a new document cannot attend to the previous one's
  history. A chunked run and a whole-sequence run agree to 1e-4.
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
aleph/core/core.py        the square recurrent core, branches, QAT
aleph/core/attention.py   temporal attention over the state history
aleph/core/plasticity.py  delta-rule fast weights at synaptic resolution
aleph/core/ponder.py      learned halting law over extra core steps
aleph/data/stream.py      tokenizers, token cache, B-cursor stream
aleph/train/trainer.py    warmup/cosine, truncated BPTT, telemetry
aleph/train/muon.py       Muon on the operators, AdamW on everything else
tools/build_corpus.py     TinyStories -> uint16 token cache
tools/experiment.py       train / smoke / ablate / eval / depends
tools/make_notebook.py    generates the Colab notebook
tests/test_aleph.py       112 tests, CPU, ~30s
```

`Research/` is gitignored: reference material, not library.

## License

MIT. See [LICENSE](LICENSE).
