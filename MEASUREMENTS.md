# Measurements

Every number here was produced by a script in `tmp/` or `tools/` on an RTX
3060 Ti (8 GiB, sm_86), torch 2.7.1+cu118. Where a measurement overturned a
design decision, the decision it overturned is named — those are the entries
worth reading, because the rest only confirm what was already expected.

## The token interface

`tmp/probe_codebook.py`, `tmp/probe_coherence.py`.

A frozen sparse-ternary codebook replaces a learned embedding table and output
head. The question was whether `V` codes stay separable in `N` dimensions and
whether a linear map can hit one well enough to be decoded.

| V | N | trainable | worst-case overlap | rms overlap | N^-1/2 |
|---|---|---|---|---|---|
| 2048 | 512 | 2,561 | 0.2500 | 0.0442 | 0.0442 |
| 4096 | 512 | 4,609 | 0.2812 | 0.0442 | 0.0442 |
| 16384 | 1024 | 17,409 | 0.2500 | 0.0312 | 0.0312 |

Decoding is exact: top-1 = 1.000 with margin 0.74–0.84 at every size and
density tested.

**Overturned:** the docstring originally claimed sparsity costs nothing because
the deviation depends on `N` alone. The *rms* does — 0.0442 at every density
from `nnz=8` to `nnz=512` — but the *worst case* is governed by `nnz`
(0.50 at 8, 0.28 at 32, 0.22 at 128, then flat). Density is free in parameters
and in time (168 µs vs 166 µs per readout at `nnz` 32 vs 512), so the default
moved to 64, where the readout margin stops improving.

### What the interface collects before any training

At `V=4096`, zipfian counts, `N=512`:

| | nats | bits | cost |
|---|---|---|---|
| uniform over V | 8.3178 | 12.001 | — |
| frozen codes + unigram bias | 5.6239 | 8.114 | 4,609 params |
| true entropy of the distribution | 5.5992 | 8.078 | — |

Essentially optimal, 3.9 bits below uniform. A learned embed+head would cost
4,194,304 parameters for the same interface: **910× more**.

This also sets the floor every run is read against. On the TinyStories cache
the unigram-bias-only loss is **6.1097 nats = 2.2376 bpb** — a run at or above
that has learned nothing, whatever it looks like next to `ln(V)`.

## Throughput

`tmp/probe_speed.py`. N=512, seq=256, depth=4.

| batch | ms | tok/s | vs batch 1 |
|---|---|---|---|
| 1 | 245.65 | 1,042 | 1.00× |
| 16 | 268.15 | 15,275 | 1.09× |
| 128 | 282.85 | 115,851 | 1.15× |

128× the work for 1.15× the time: the clock tracks kernel launches, not
arithmetic. Batch is nearly free and should be maximal.

`torch.compile` on the step measured **0.88× — a slowdown** (274.90 ms eager
against 310.68 ms compiled). Inductor cannot fuse across a data-dependent
Python loop and the per-step graph is too small to win alone.

### Depth, before and after the rewrite

**Overturned:** `depth` was implemented as a chained refinement stack and
documented as "nearly free in wall-clock" from reasoning. Measured, it cost
4.59× the time for 2.0× the parameters — `3·depth` sequential launches in a
loop that is already launch-bound.

Rewritten as parallel branches concatenated into one wide matmul pair:

| depth | ms | params | vs d=0 | param ratio |
|---|---|---|---|---|
| 0 | 55.16 | 268,289 | 1.00× | 1.00× |
| 2 | 109.61 | 399,873 | 1.99× | 1.49× |
| 4 | 158.15 | 530,945 | 2.87× | 1.98× |
| 8 | 151.26 | 793,089 | 2.74× | 2.96× |
| 16 | 154.66 | 1,317,377 | 2.80× | 4.91× |

Time **saturates at 2.8×** while parameters keep climbing. Past depth 4,
capacity is genuinely free — which is why the default is 8.

### Training step

batch 64, seq 256, N=512, depth=4:

| | ms | tok/s | peak VRAM |
|---|---|---|---|
| no checkpointing | 973.6 | 16,828 | 1,937.9 MiB |
| gradient checkpointing | 1,655.3 | 9,898 | 1,424.7 MiB |

## The backward pass is where this architecture bites

`tmp/probe_gradnorm.py`, `tmp/probe_jacobian.py`. The most important
measurement in the project.

The first real run held `clip% ≈ 100` with `|g| ≈ 4–5`. Looking for why:

| seq | depth=0 | depth=8, V=0 | depth=8, V=0.05 |
|---|---|---|---|
| 32 | 12.88 | 125.90 | 49.9 |
| 64 | 33.26 | 391.20 | 177.8 |
| 128 | 522.97 | 5,085.42 | 1.577e4 |
| 256 | 151,091.02 | 1,473,029.12 | 2.902e8 |

**289× per doubling, and identical at `depth=0`.**

**Overturned, twice.** The first diagnosis was that the refinement branches
compound — 99.4% of the gradient norm sits in `U/V`, which looked conclusive.
It was wrong: `depth=0` explodes at exactly the same rate, so the branches
merely carry the gradient rather than create it. The second guess, residual
scaling by `1/depth`, measured 1,691× per doubling against 18,408× — a
reduction that changes nothing about the exponent.

The per-step Jacobian says why: **1.0647 at depth=0**. Slightly above 1,
compounded 256 times, is `10^7`. Per-step RMSNorm pins the *state* to `√N`
exactly — measured 15.999998 at N=256 — but its Jacobian still passes gradient,
so the forward bound buys nothing in the backward pass. The core docstring
claimed explosion was "structurally impossible"; that claim was true of the
state and false of the gradient, and has been corrected.

### The fix: a backward window

Detaching the state every `window` steps, seq=256:

| window | &#124;g&#124; |
|---|---|
| full | 1.122e8 |
| 8 | 2.418 |
| 16 | 4.107 |
| 32 | 12.32 |
| 64 | 173.7 |
| 128 | 1.678e4 |

The exponential is visible in the window column itself. The forward pass is
bit-identical at any window — every position is still predicted and still
supervised — so only gradient reach is cut. Because `W` is one shared matrix,
what is learned inside a window applies at every step.

## Clipping

`tmp/probe_clip.py`. 500 steps, batch 64, seq 256, N=512, depth=8, seed 0.

| clip | lr | final loss | bpb | clip% |
|---|---|---|---|---|
| 1 | 1e-3 | 5.0772 | **1.8593** | 90% |
| 10 | 1e-3 | 5.2315 | 1.9158 | 0% |
| 1000 | 1e-3 | 5.2769 | 1.9324 | 0% |
| 10 | 3e-3 | 5.6409 | 2.0657 | 0% |

**Overturned:** `clip% = 100` was read as the clip throttling the schedule —
`lr·clip/|g|` is a quarter of the scheduled rate at `|g|≈4`. Relaxing it makes
things *worse* at every value tried. The aggressive clip is acting as a
stabilizer, not a bottleneck, which is a known property of clipping on
recurrent models. `clip=1.0, lr=1e-3` stands.

`lr=3e-3` is worse again, consistent with the overfit probe where `depth=4`
diverged at 3e-3 and reached 0.0192 nats at 1e-3.

## Learning rate and depth interact

`tmp/probe_stability.py`. 300 steps overfitting one 16×64 batch.

| depth | lr | min loss | final | last-50 spread |
|---|---|---|---|---|
| 0 | 3e-3 | 0.0626 | 0.0626 | 0.0165 |
| 0 | 1e-3 | 0.1090 | 0.1090 | 0.0325 |
| 4 | 3e-3 | 0.1871 | 1.0458 | 2.7085 |
| 4 | 1e-3 | **0.0192** | 0.0192 | 0.0214 |

`depth=4` at the right rate is 3× better than `depth=0` at its best. At the
wrong rate it is erratic, which initially read as depth being harmful. Forward
scale through the branch stack grows only 1.04–1.08×, so this is step-size
tolerance rather than an explosion — hence warmup, and hence a depth comparison
at one fixed rate measures the schedule.

## Corpus

TinyStories, 400,000 stories, in-domain BPE:

- **92,012,977 train / 929,424 val tokens**, V=4096, 3.94 chars/token
- BPE fit on 60k stories in 192 s; full encode ~11 min
- A 1M-token budget is 0.011 epochs of this split

Every bits-per-byte figure above is defined against this cache. A rebuilt cache
moves the held-out boundary, so a comparison across caches is not a comparison.
