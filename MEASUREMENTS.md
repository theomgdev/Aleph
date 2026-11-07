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

## The first full run

`tools/experiment.py --mode train --neurons 512 --depth 8 --batch 96 --seq 256
--window 32 --steps 6000 --lr 1e-3 --warmup 300`, on the 92M-token TinyStories
cache (3.94 chars/token).

| | |
|---|---|
| trainable | **793,089** — core 261,632 + refine 524,288 + interface 4,609 + mod 2,560 |
| final val loss | 3.2498 nats |
| final val bpb | **1.1901** |
| perplexity | 25.79 |
| data | 147.5M tokens, 1.60 epochs |
| throughput | 38,921 tok/s |

Against the unigram floor of 2.2376 bpb, the core earned **1.047 bpb**. Train
and val tracked within 0.02 the whole way (1.177 train against 1.1984 val at
step 6000), so nothing here is overfitting — at 1.6 epochs the model is still
data-limited rather than capacity-limited, and the val curve was still falling
when the schedule ended.

Progression: 2.163 bpb (step 250) → 1.4900 (1500) → 1.3027 (3000) → 1.2528
(3750, one epoch) → 1.2265 (4500) → **1.1901** (6000).

`|h|` drifted 22.30 → 16.97 as the output RMSNorm gain learned to shrink the
state; `|g|` fell 4.23 → 2.05, and `clip%` held at 100 throughout, which the
clip sweep above establishes is the intended regime rather than a fault.

### Samples

Unprompted continuations at temperature 0.8, top-k 40:

> `Once upon a time` → *" with a big, soft truck. They had a lot of fun with
> their toys. [...] Tim and Sam are friends. They like to play with their toys,
> but they do not like to play with Anna."*

> `The little girl` → *"'s friend, a big, red bird. The cat was very proud of
> her. [...] Sara and Tom are friends. They like to play with toys."*

Grammatical sentences, consistent characters across clauses, and correct story
structure including paragraph breaks. Semantic coherence over a whole story is
not there — "a big, soft truck", and the third sample drifts — which is what
1.6 epochs at this size buys.

### What this is and is not evidence for

It is evidence that a frozen `V + N + 1` token interface can carry a language
model: the readout was never trained, and the model produces English through it.

It is not yet a parameter-efficiency record, and the comparison that would
decide that has not been run. The nearest published reference is
TinyStories-1M (Eldan & Li 2023, arXiv:2305.07759) at 0.40M non-embedding
parameters against Aleph's 0.79M core+refine, scored by GPT-4 rubric rather
than bits per byte — so the two numbers are not comparable as they stand, and
saying otherwise would be the exact error `README.md` warns about.

## The synthesis

Four mechanisms, each measured on its own at 400 steps and 16,384 tokens per
step from seed 0 (6.55M tokens), `N=512, depth=8, window=32`. Every optional
branch is zero-initialized at its output projection, so all arms share an
identical loss at step 0 and the difference is the mechanism rather than the
initialization.

| arm | params | train | val nats | val bpb | Δ nats | tok/s |
|---|---|---|---|---|---|---|
| AdamW, `B=64, T=256` | 793,089 | 5.2518 | 5.3332 | 1.9530 | — | 30,935 |
| AdamW, `B=256, T=64` | 793,089 | 5.2518 | 5.2949 | 1.9390 | −0.038 | 112,128 |
| + Muon | 793,089 | 4.6852 | 4.7643 | 1.7447 | −0.531 | 105,388 |
| + aux encoder/readout | 861,185 | 4.4233 | 4.5074 | 1.6506 | −0.257 | 101,709 |
| + temporal attention | 957,505 | 4.6025 | 4.6999 | 1.7211 | −0.064 | 90,767 |
| + delta plasticity | 961,665 | 4.5694 | 4.6452 | 1.7011 | −0.119 | 76,473 |
| all four | 1,194,177 | 4.3030 | 4.4094 | 1.6147 | −0.885 | 78,119 |
| all four, compact | 1,027,169 | 4.3354 | 4.4243 | 1.6202 | −0.871 | 72,070 |

Each Δ is against the `B=256, T=64` Muon arm for the three branches, and
against AdamW for Muon itself. The compact row halves attention's `head_dim`
(32 → 16) and quarters plasticity's `dim` (64 → 32): −167k parameters for
+0.015 nats, so compact is the better point and is the default.

The integrated library reproduces the compact row at 4.4179 nats — 0.006 below
the probe, because constructing the branches inside `__init__` draws RNG in a
different order than bolting them on afterwards.

### Geometry: wide and short is free

The state and both branch caches cross forward calls, so `seq` is a launch
granularity rather than a context length — a token sees unbroken history and
credit reaches `window` tokens back either way. At a fixed 16,384 tokens per
step and `window=32`:

| batch | seq | ms/step | tok/s | peak VRAM |
|---|---|---|---|---|
| 64 | 256 | 324.8 | 50,439 | 1,660.4 MiB |
| 128 | 128 | 169.4 | 96,716 | 1,660.0 MiB |
| 256 | 64 | 103.6 | 158,085 | 1,659.9 MiB |
| 512 | 32 | 72.4 | 226,429 | 1,659.8 MiB |

Identical token count, identical credit horizon, identical memory, 4.5× the
throughput. The mechanism is the launch bound: 16,384 tokens at `T=256` is 256
sequential step launches, at `T=64` it is 64 launches over 4× wider matrices.
The loss moved the right way too (1.9530 → 1.9390), because a step averages
over 256 independent stream positions instead of 64.

This is the largest speedup in the project and it has no cost, which is why
`(256, 64)` is the default geometry.

### Where the time goes

At `B=64, T=256` the recurrence is 308.3 ms of a 314.3 ms step. The readout and
cross-entropy over `V=4096` — 206 GFLOPs against the recurrence's 25.7 — take
19.6 ms in fp32 and 14.9 in bf16. So 98% of the time is spent on 11% of the
arithmetic: the loop is latency-bound, not compute-bound, and the only thing
that matters is how many sequential launches it issues.

### Quantization-aware training

QAT on `W`, `U`, `V` with a straight-through estimator, measured on the compact
synthesis at 400 steps:

| precision | val nats | val bpb | Δ nats |
|---|---|---|---|
| fp32/tf32 | 4.4243 | 1.6202 | — |
| int8 | 4.4383 | 1.6253 | +0.014 |
| int4 | 4.6913 | 1.7180 | +0.267 |
| ternary (1.58-bit) | 4.9016 | 1.7950 | +0.477 |

int8 is free — 4× smaller weight storage for a difference inside run-to-run
noise. Below that the core pays real capacity. The ternary penalty of +0.477
nats independently reproduces the published expectation (+0.50 for sub-200M
models, `Research/22-quantization-and-qat.md`), so it is recorded as a negative
result: this architecture at this scale does not support ternary weights, and
no configuration was sought that would make it pass.

The STE needs both halves detached to be correct, which is a bug worth
recording because nothing downstream would have revealed it. Writing
`q + (w - w.detach())` rather than `w + (q - w).detach()` makes the forward
value exactly `q` instead of one ulp away — so an exported weight is literally
ternary. But `q` must *also* be detached: `scale` is a function of `w`, so a
live `q` sends gradient through the quantizer's own scale, and the measured
per-weight gradient came back in [0.69, 1.31] instead of 1.0. Both runs would
have trained and neither loss curve would have looked wrong.

### Muon, reconsidered

An earlier probe put Muon behind AdamW (2.0633 against 1.9371 bpb). That probe
had `set_lr` overwriting `muon_lr`, so the Muon group was running at the AdamW
rate. With that fixed and the RMS matching at `0.2·sqrt(max(A,B))`, Muon leads
by 0.531 nats at zero parameter cost — the largest single win in the table.

The advantage is in convergence *rate*, not in the final optimum, so it is
invisible early: at 25 steps AdamW is still ahead by 0.20 nats. A probe short
enough to be cheap is long enough to reach the wrong conclusion here.
