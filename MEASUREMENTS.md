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

## Are the branches memories, or just parameters?

The ablation says plasticity is worth 0.119 nats and attention 0.064. It does
not say why, and two explanations fit either number: the branch carries
information across time, or its projections are simply more capacity. Those
predict different things about a trained model, so intervening on the state of
a finished checkpoint separates them. Measured on the step-2000 checkpoint
(3.1844 nats, 1.1661 bpb, 40 val batches at `B=32, T=256`):

| intervention | nats | bpb | Δ nats |
|---|---|---|---|
| none | 3.1844 | 1.1661 | — |
| plasticity branch zeroed | 3.6412 | 1.3334 | +0.457 |
| attention branch zeroed | 6.4918 | 2.3773 | +3.307 |
| attention cache truncated to 1 | 3.2428 | 1.1875 | +0.058 |
| attention cache truncated to 8 | 3.2191 | 1.1789 | +0.035 |
| attention cache truncated to 32 | 3.1875 | 1.1673 | +0.003 |
| attention cache truncated to 64 | 3.1794 | 1.1643 | −0.005 |
| aux encoder zeroed | 5.3063 | 1.9432 | +2.122 |
| aux readout adapter zeroed | 4.8359 | 1.7709 | +1.652 |

Every branch is load-bearing by the end of training, and by far more than its
ablation delta — removing attention from a trained model costs 3.3 nats where
training without it cost 0.064. Those are different quantities: the ablation
measures what a mechanism adds when the rest of the model can compensate, and
this measures what the trained model has come to depend on. A branch that looks
marginal in an ablation can still be load-bearing once the weights have
organized around it, and that is not a contradiction.

Varying how long plasticity's trace is allowed to persist answers the original
question directly:

| trace horizon | nats | bpb |
|---|---|---|
| 2 tokens | 3.5616 | 1.3043 |
| 4 | 3.4772 | 1.2734 |
| 16 | 3.2350 | 1.1847 |
| 64 | 3.1119 | 1.1396 |
| full 256 | 3.0670 | 1.1232 |

Monotone in the horizon and still improving at 256, so the trace is a memory
rather than extra capacity, and it is using every token it is given. Attention
saturates at a cache of 32. The two branches are therefore reading different
timescales — attention a short recent window, plasticity an unbounded running
association — which is the mechanism behind their stacking near-additively.

One row deserves its own note, because it looks like a bug and is not.
Resetting the trace before every token scores *identically* to deleting the
branch (3.6412 both). That is correct arithmetic: the delta rule reads before it
writes, so a trace reset to zero is read as zero, and `o_proj` is bias-free, so
the branch output is identically zero at every position. The horizon sweep above
is the right form of the experiment.

## The decode bug, and why the score did not show it

The first synthesis run reached 1.0016 bpb — a large improvement on the 1.1901
baseline — and generated markedly *worse* text than the baseline had. Those two
facts cannot both describe the same model, and the contradiction is the only
reason the bug was found: every aggregate number was better, every test passed,
and the samples were the sole dissenting evidence.

The cause was the auxiliary encoder. `nn.Conv1d(kernel_size=4, padding=3)` with
the output sliced to `[..., :t]` is correctly causal within one call, but it
holds no state across calls, so its left context is zeros at the start of every
call. Training called it once per 64-token chunk, where 3 of 64 positions are
affected. Generation calls it once per token, where *every* position is
affected: each token was convolved against three zeros instead of its three
real predecessors. The model was scored with 4-gram context and sampled without
it.

Measured on real validation text at `T=24`, comparing a whole-sequence forward
against the same tokens decoded one at a time:

| configuration | max abs logit diff | argmax agreement |
|---|---|---|
| all branches | 1.2952 | 89.6% |
| without the aux encoder | 0.0006 | 100.0% |
| without attention | 0.4467 | 91.7% |
| without plasticity | 1.2391 | 95.8% |
| core only | 0.0010 | 100.0% |

Isolating by removal is what identifies the culprit: dropping the encoder is
the only intervention that restores agreement, so the other branches' caches
were carrying correctly. Ten percent of generated tokens were drawn from the
wrong distribution.

The fix carries the last `kernel-1` code vectors across calls as a detached
tail, exactly as the hidden state and the other caches do, and removes the
conv's own padding so the two cannot double-count. After it, the same
comparison gives 2.65e-06 with 100% argmax agreement, and an 8-token chunking
gives 1.19e-06.

Causality was then checked separately, because a consistency fix is exactly the
kind of change that can buy agreement by letting the conv see the future.
Editing the token at position `k` moves nothing at any position before `k`
(difference exactly 0.0 at `k = 4, 8, 12`) and everything from `k` onward.

Three things are worth keeping from this.

A zero-initialized branch cannot be tested by a freshly built model: the branch
contributes nothing, and a broken branch contributes nothing just as
convincingly. The consistency tests now run against a model whose branch
outputs have been randomized to non-zero, which is what training produces.

Chunk-invariance is not a nicety here, it is the correctness condition. Any
module that sees a window of the sequence must carry that window across calls,
or the chunk size silently becomes a hyperparameter of the model's output —
and generation is the limiting chunk size of 1, so a bug of this shape always
lands on the samples and never on the loss.

And the samples earned their place in the report. No aggregate caught this.

## Why the samples lag the score

After the decode fix the generated text improved but still dropped function
words and inserted commas at every temperature, including greedy — so this is
not a sampling artifact. Two measurements locate it.

**Sampling temperature was wrong for this model.** Teacher-forced on validation
text, the predictive distribution has mean entropy 3.27 nats and mean top-1
probability 0.390. That is a genuinely uncertain distribution, and `T=0.8` on
top of it is aggressive. Free-running entropy stays flat across 70 tokens
(3.2–4.0 nats, no upward drift), so the model is not degenerating — the sampler
was just drawing from the tail. At `T=0.6` the same checkpoint produces *"in a
small house, there, a little girl named Lily. She loved to play with her
blocks."* The default is now 0.6; 0.8 is a knob, not a given.

**And half the logit signal is a constant.** Decomposing the readout
`h @ Hᵀ · out_scale + bias` on real text at step 2,000:

| term | std across the vocabulary |
|---|---|
| state-dependent, `h @ Hᵀ · out_scale` | 2.5893 |
| fixed unigram bias | 2.5439 |
| ratio | 1.018 |

The bias is one vector, identical at every position, and it contributes as much
logit variance as the entire recurrence, attention, plasticity and adapter
combined. That is consistent with both observations rather than contradicting
either: the unigram prior is worth 3.9 bits, so it genuinely improves bits per
byte, while at any single position the context-dependent part has only half the
say — which is exactly how a frequent token wins a slot that grammar had
already claimed.

It is also a caution about this project's own thesis. The frozen interface
removed a 4.2M-parameter dictionary, and a 4,096-entry logit bias is a much
smaller dictionary doing a smaller version of the same job. Those 4,609
interface parameters are honestly counted as trainable, but anyone reading the
bits-per-byte figure as a measure of *computation* should know that about half
the readout's variance at this checkpoint is a corpus statistic rather than
something the core computed. The `--no-unigram` arm exists to measure that, and
quoting it beside the headline number is the honest form.

## The cold-start cliff

After the decode fix and the temperature fix, the step-6000 samples had a
distinctive shape: the first ~8 tokens after each prompt were garbled, and
then the model recovered into clean prose (*"The girl was so sad and she
realized that she was alone, but she never forgot her special gift"*). The
prompts are 3–4 tokens.

Per-position loss on validation text after a cold start (`state=None`), averaged
over 768 sequences:

| position band | nats | bpb |
|---|---|---|
| 0 | 11.5401 | 4.2260 |
| 1 | 10.4133 | 3.8134 |
| 2–3 | 7.6042 | 2.7847 |
| 4–7 | 5.2395 | 1.9187 |
| 8–15 | 3.9707 | 1.4541 |
| 16–31 | 3.2281 | 1.1821 |
| 32–63 | 2.8757 | 1.0531 |
| 64–127 | 2.7400 | 1.0034 |

Position 0 is **worse than uniform** (8.32 nats at `V=4096`), and the model does
not come within 0.05 nats of its warm score until position 38. A 4-token prompt
leaves generation at position 4, where the loss is still 5.24 nats — so the
first generated tokens are drawn from a barely warmed-up state, and the prose
only becomes good once the model's own outputs have warmed it up.

The mechanism is a training-vs-use mismatch created by our own speedup.
Carrying state across calls is what made the short-chunk geometry free, and it
also meant step 0 of 6,000 was the only cold start the model ever saw — ~0.02%
of training positions, against 100% of generation's first token. The model was
never trained to begin.

Two fixes were tested at 1,200 steps (`B=256, T=64`), alone and together:
`reset_every=16` (start a fresh sequence every 16 steps, state and all branch
caches) and a learned initial state `h0` (`N` parameters). Mean nats by position
band after a cold start, plus the position where the loss comes within 0.05 of
warm:

| arm | pos 0 | pos 1 | 2–3 | 4–7 | 8–15 | 16–31 | 32–63 | 64–127 | recovery |
|---|---|---|---|---|---|---|---|---|---|
| zeros `h0`, no reset | 7.44 | 7.47 | 7.35 | 5.52 | 4.39 | 4.01 | 3.76 | 3.72 | 37 |
| zeros `h0`, reset 16 | 5.03 | 5.30 | 4.44 | 4.17 | 3.88 | 3.80 | 3.69 | 3.69 | **10** |
| learned `h0`, no reset | 8.79 | 7.94 | 6.99 | 5.38 | 4.28 | 3.93 | 3.74 | 3.71 | 22 |
| learned `h0`, reset 16 | 5.67 | 4.80 | 4.36 | 4.17 | 3.87 | 3.81 | 3.70 | 3.70 | **10** |

`reset_every=16` is the whole win. It cuts the recovery horizon from 37 tokens
to 10, drops position 2–3 from 7.35 to 4.44, and slightly *improves* the warm
score too (3.72 → 3.69), because 16 steps of 64 tokens is 1,024 tokens — longer
than any dependency the model was using.

`learned h0` did not beat `zeros h0` once resets were on (5.67 vs 5.03 at
position 0, identical recovery of 10, 0.01 worse warm), so it was removed. The
problem was never that zero is a bad starting point; it was that the starting
point never got gradient. Zero parameters beat 512.

One hazard surfaced while testing `h0` and is worth recording even though the
parameter was dropped: passing a zero-initialized vector through `RMSNorm` gave
`|g| = 6,649` on `h0` at step 0, because the norm divides by the input's own
RMS and its Jacobian diverges at the origin. And placing an RNG-drawn parameter
after an `if depth > 0:` block shifts the generator state by `depth`, which
broke `test_depth_is_inert_at_init` on its first run.

## The full synthesis run

6,000 steps at the default geometry and schedule (`N=512, depth=8, B=256, T=64,
window=32, reset_every=16, lr=1e-3, warmup=200`, Muon on operators, all three
branches on in the compact configuration), against the same cache as the
baseline (`tinystories400k_bpe4096_b0b0cc61.bin`, 92,012,977 train / 929,424
val tokens, 3.94 chars/token):

| run | trainable params | tokens | tok/s | best val nats | best val bpb | final val bpb | ppl |
|---|---|---|---|---|---|---|---|
| baseline (`core + refine`, AdamW, `B=96, T=256`) | 793,089 | 147.5M (1.60 ep) | 38,921 | 3.2498 | 1.1901 | 1.1901 | 25.79 |
| synthesis (all four, `B=256, T=64, reset=16`) | 1,027,169 | 98.3M (1.07 ep) | 58,709 | **2.7126** | **0.9934** | **0.9994** | **15.07** |

−0.537 nats (−0.197 bpb, perplexity 25.79 → 15.07) in two-thirds the token
budget and at 1.5× the throughput, for +234k parameters. The synthesis crosses
the baseline's final 6,000-step score before step 2,000.

### Trajectory

| step | train nats | val nats | val bpb | val ppl | `|g|` | `|h|` | epochs |
|---|---|---|---|---|---|---|---|
| 500 | 4.2929 | 4.3320 | 1.5864 | 76.10 | 11.15 | 21.51 | 0.09 |
| 1,000 | 3.5416 | 3.6305 | 1.3295 | 37.73 | 6.53 | 21.20 | 0.18 |
| 1,500 | 3.2286 | 3.3860 | 1.2400 | 29.55 | 4.62 | 20.98 | 0.27 |
| 2,000 | 3.0598 | 3.1806 | 1.1648 | 24.06 | 3.62 | 20.77 | 0.36 |
| 3,000 | 2.8977 | 2.9317 | 1.0736 | 18.76 | 3.04 | 20.57 | 0.54 |
| 4,000 | 2.8055 | 2.8012 | 1.0258 | 16.46 | 2.81 | 20.49 | 0.71 |
| 5,000 | 2.7112 | 2.7540 | 1.0085 | 15.71 | 2.70 | 20.46 | 0.89 |
| 6,000 | 2.7010 | 2.7126 | 0.9934 | 15.07 | 2.67 | 20.46 | 1.07 |

### Samples

Unprompted continuations at `T=0.6`, top-k 40, after all three fixes (carried
conv tail, temperature, `reset_every=16`):

> `Once upon a time` → *", there was a little boy named Timmy. Timmy loved to
> play outside and explore the world around him. One day, Timmy saw a big,
> scary monster called Billy. Billy was scared and wanted to catch it.
> \n\n\"Hi, Timmy! I'm scared of the monster,\" said Billy."*

> `The little girl` → *" was very sad and her mom was happy to help.\n\nThe
> girl thanked her mom for helping her and went home."*

> `Tom and Sara` → *" looked at the ground. It liked to play with it. Sara
> liked to play with the ball."*

Every prompt begins cleanly from its first token — no warmuptransient, dialogue
with matching quotation marks and speaker attribution, and character names that
persist across paragraphs.

### What the step-6000 checkpoint depends on

`python tools/experiment.py --mode depends --tag main` (40 val batches at
`B=32, T=256`):

| intervention | nats | bpb | Δ nats |
|---|---|---|---|
| none | 2.7905 | 1.0219 | — |
| attention branch zeroed | 10.1966 | 3.7341 | +7.406 |
| plasticity branch zeroed | 3.2568 | 1.1927 | +0.466 |
| aux encoder zeroed | 5.8860 | 2.1555 | +3.096 |
| aux readout adapter zeroed | 5.2516 | 1.9232 | +2.461 |
| unigram logit bias zeroed | 4.4007 | 1.6116 | +1.610 |
| attention cache limited to 1 | 2.9190 | 1.0689 | +0.129 |
| attention cache limited to 8 | 2.8563 | 1.0460 | +0.066 |
| attention cache limited to 32 | 2.7674 | 1.0134 | −0.023 |
| attention cache limited to 64 | 2.7419 | 1.0041 | −0.049 |
| plasticity trace limited to 2 | 3.3645 | 1.2321 | +0.574 |
| plasticity trace limited to 16 | 3.0038 | 1.1000 | +0.213 |
| plasticity trace limited to 64 | 2.8209 | 1.0330 | +0.030 |

Readout decomposition at step 6,000: state-dependent std 2.8344, fixed bias std
2.6528, `signal/bias = 1.068`.

One number in that table points at the next change rather than confirming the
last one: truncating the attention cache from 128 to 64 *lowers* the loss by
0.049 nats, and did the same at steps 2,000 (−0.005) and 3,500 (−0.040). The
effect grows with training. So keys older than ~64 tokens are net noise for the
attention branch while plasticity continues to improve out to 256 — further
evidence that the two branches have divided the timescales between them, and
that `attn_window` is wider than the job attention has taken on.

## Thinking longer per token, as extra core steps

Depth here is time, so "spend more computation on a hard token" has an obvious
form: apply the step more than once between two tokens, with a frozen code on
the input marking that no token arrived. The wait code lives outside `H`, so it
is injected and never predicted — a code the readout cannot emit is one the
softmax cannot be trained to suppress, which it otherwise would be, since every
supervised position pushes every non-target logit down.

The mechanism works. At 800 matched steps, `N=512`, depth 8, batch 256, seq 64,
window 32, seed 0, for **zero added parameters**:

| arm | params | steps | val nats | bpb | sec | vs base |
|---|---|---|---|---|---|---|
| `wait=0` | 1,027,169 | 800 | 4.0530 | 1.4842 | 147 | — |
| `wait=1` | 1,027,169 | 800 | 3.9193 | 1.4353 | 210 | −0.134 |
| `wait=2` | 1,027,169 | 800 | 3.8909 | 1.4249 | 257 | −0.162 |
| `wait=4` | 1,027,169 | 800 | 3.9316 | 1.4398 | 425 | −0.121 |
| `depth=24 wait=0` | 2,075,745 | 800 | 3.8131 | 1.3964 | 157 | −0.240 |
| `wait=0` @ `wait=2` seconds | 1,027,169 | 1,395 | 3.6069 | 1.3209 | 266 | −0.446 |
| `wait=0` @ `wait=4` seconds | 1,027,169 | 2,304 | 3.2496 | 1.1900 | 427 | −0.803 |

And it loses anyway. The first four rows are the measurement the idea predicts:
extra steps buy real loss for no parameters, and the curve turns at `wait=4`.
The last two rows are the measurement that decides it. `wait=k` costs `k+1`
sequential launches per token on a loop that is launch-bound, and a baseline
given those same seconds — more steps, more data, same parameters — beats the
waiting model by 0.284 nats at matched 266s and by 0.682 nats at matched 427s.
The gap widens with budget, so it is not a warmup artifact.

Two mechanisms explain the shape. The turn at `wait=4` is the backward pass:
`wait=k` multiplies the effective backward length by `k+1`, so `window=32`
becomes 160 steps at `wait=4`, and the per-step Jacobian sitting above 1 does
what it always does — `|g|` hit 722.99 at step 200 there against 7.93 at
`wait=1`. And `depth=24` reaching −0.240 in 157s says the useful part is extra
nonlinear capacity per token, not elapsed recurrence: `depth` buys it inside one
launch, where arithmetic is nearly free, while `wait` buys it in sequential
launches, which is the one resource this loop does not have.

So `wait` stays in the library as a measured axis (`--wait`, default 0, inert at
0) and does not become a default.

### What about learning the allocation?

The objection to the fixed table is fair: `wait=4` forces `the` and `and` to pay
what a hard referent pays, which is the worst possible spending pattern. An
adaptive halting law (`Ponder`, `N+1 = 513` parameters, PonderNet's survival
distribution over candidates $h_0, \dots, h_M$, exiting as the norm-rescaled
expectation $\sum_k p_k h_k$ with a KL toward a truncated geometric prior) was
therefore built and measured on the same 800-step configuration:

| arm | params | mean wait | val nats | bpb | vs `wait=0` |
|---|---|---|---|---|---|
| `wait=0` | 1,027,169 | 0.00 | 4.0688 | 1.4900 | — |
| `fixed wait=1` | 1,027,169 | 1.00 | 3.9096 | 1.4317 | −0.159 |
| `fixed wait=2` | 1,027,169 | 2.00 | 3.9319 | 1.4399 | −0.137 |
| `ponder` (`prior=0.5`, $M=8$) | 1,027,682 | 2.63 | **3.7680** | **1.3799** | **−0.301** |
| `ponder` (`prior=0.33`, $M=8$) | 1,027,682 | 4.11 | 3.8041 | 1.3931 | −0.265 |

On tokens alone, `ponder` beats every fixed arm by another 0.14 nats for 513
parameters — and even solves the backward-pass turn that hurt `wait=2` and
`wait=4`, because $p_0 h_0$ keeps a direct skip path around the deeper steps.

Then the diagnostic asks what the head actually learned (`tmp/probe_ponder_why.py`,
65,536 validation tokens):

* **Per-token difficulty varies by 8.83 nats** (p10 = 0.53, p90 = 9.36), so there
  is plenty to allocate against.
* **Per-token expected steps barely move**: under per-token KL (`mean=2.634`,
  `sd=0.020` across batches, `0.32` per token; freezing `w=0` at the same mean
  costs +0.131 nats) and under batch-marginal KL (`mean=1.687`, `sd=0.068`,
  range `[1.21, 1.79]`; freezing `w=0` costs +0.033 nats).
* **Why**: two structural reasons. First, 87.0% of the energy of $h$ sits in a
  single constant cone (`|E[h]| = 19.09` out of `|h| = 20.46`, because `GELU`
  lives in the positive orthant and `RMSNorm` does not center), so a linear
  probe on $h$ spends most of its dynamic range on the constant offset. Second,
  and deeper: $\sum_k p_k h_k$ is a **depth ensemble**. Averaging $h_0, h_1,
  \dots, h_M$ with interior weights gives every token a multi-step residual
  combination of higher effective rank than any single $h_k$ alone; committing
  to one $k$ throws the other $M$ directions away.
* **The early-exit test confirms it.** Truncating the tail at eval time when the
  survival probability drops below `eps` ( trained at 200 steps, `val = 5.3866`
  at `eps=0.01`): `eps=0.20` degrades to **5.8166** (+0.43 nats) and `eps=0.40`
  to **6.7349** (+1.35 nats). Dropping the low-probability tail costs more than
  the entire mechanism gained, because the model was using the tail as part of
  its state representation rather than as an unused backup step.

So differentiable halting by state expectation does not buy "stop early on easy
tokens" — it buys a learned convex combination of $M{+}1$ sequential depths,
which must all be run to produce the vector the readout was trained to read.

### Mixing the loss instead of the state

That conclusion is a property of mixing *states*, and the choice was made on a
memory argument rather than a measured one, so it had to be checked. PonderNet
mixes losses: $\mathcal{L} = \sum_k p_k \,\text{CE}(y, \text{readout}(h_k))$,
which supervises every candidate on its own, so halting at $k$ and decoding
from $h_k$ is exactly what training rehearsed. Both rules, one seed, one axis
changed ($N{=}256$, depth 4, $M{=}4$, $B{=}128, T{=}64$, 600 steps,
`tmp/probe_lossmix.py`), scoring each candidate's own readout:

| | mixture CE | $h_0$ alone | $h_1$ | $h_2$ | $h_3$ | $h_4$ | mean steps |
|---|---|---|---|---|---|---|---|
| state-mixing | 6.0019 | 6.1699 (**+0.168**) | 6.5632 | 7.1007 | 7.1825 | 7.4375 | 0.69 |
| loss-mixing | 6.0321 | 6.0273 (**−0.005**) | 6.5991 | 6.9352 | 7.0218 | 7.0761 | 0.04 |

Loss-mixing does make the early exit free — $h_0$ decodes as well alone as the
mixture does, against +0.168 nats under state-mixing, which is the same defect
the $\epsilon$-truncation test found at $N{=}512$. The structural prediction
held exactly.

**And it reveals that there was nothing to allocate.** Under *both* rules the
candidates get monotonically *worse* with depth ($h_0 \to h_4$: 6.17→7.44 and
6.03→7.08). Extra applications of $W$ on a constant code do not refine a state
toward a better prediction — they walk it away from one. So the −0.301 nats the
mixture earned was an **ensemble** of $M{+}1$ differently-degraded views, which
is a real effect and an honest one, but it is not deliberation and it cannot be
made cheap: an ensemble needs its members. Given a readout per candidate and
therefore an honest choice, the head priced the steps at **0.04** and declined
to think.

The memory figure in the rejected argument was also measured rather than left
as an estimate (`tmp/probe_readout_cost.py`, $B{=}256, T{=}64, V{=}4096,
N{=}512, M{=}8$): peak allocation 5,689 MiB for loss-mixing against 1,617 MiB
for state-mixing, 3.5× and +4.0 GiB. The cost was real. What was wrong was
letting it settle an architectural question it only priced — the cheap side
bought its saving with the capability the feature existed for, and the
measurement that would have caught it is the one above.

Both `--wait` and `--ponder-max` remain in the library as inert axes (default 0,
bit-identical at 0), and the main model stays at one step per token. The
mechanism is in the repository because the question "does more compute per
token help here" now has a measured answer, which is better than an assumption
in either direction: **not on a constant code, and not by stopping early.**
What would have to change is the thing injected during the pause — a code that
carries information the state does not already have, rather than the same
frozen vector $M$ times.
