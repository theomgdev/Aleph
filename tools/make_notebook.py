"""Generate the Colab notebook — the deliverable.

A notebook is a JSON file, and writing it from a script rather than by hand is
what keeps it consistent with the library it drives: the cells import `aleph`
and call the same entry points `tools/experiment.py` does, so there is one
implementation and the notebook is a driver rather than a fork of it.

Structure follows the staging the run needs rather than the order a reader
might browse: environment, then corpus, then a smoke test that fails fast, then
the real run, then the ablation arms that turn it into a claim, then samples.
Every long cell prints progress on a schedule, because an unattended Colab
session that goes quiet is indistinguishable from one that has hung.
"""

import json
import pathlib

REPO = "https://github.com/theomgdev/aleph"


def md(src):
    return {"cell_type": "markdown", "metadata": {},
            "source": [line + "\n" for line in src.strip().split("\n")]}


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": [line + "\n" for line in src.strip().split("\n")]}


CELLS = [
    md("""
# Aleph — a language model that is one matrix

Aleph spends its parameters on computation rather than on a lookup table. A
conventional tiny language model at `V=4096`, `N=512` pays `2·V·N = 4.2M`
parameters for an embedding table and an output head, against `N² = 262k` for
the layer that does the work: 94% of the "model" is a dictionary, and any
parameter-efficiency figure quoted against that total is measuring the
dictionary.

Here the token interface is **frozen**. Each token gets a fixed sparse ternary
code drawn from a seeded generator, and both directions read the same table:

    inject(t) = H[t] · in_gain
    logits    = h @ Hᵀ · out_scale + bias

That costs `V + N + 1` trainable parameters instead of `2·V·N` — 4,609 against
4,194,304 at `V=4096, N=512`, **910× less** — and what it asks in exchange is
that the core learn to emit points in code space.

**Why a random code separates.** Codes are near-orthogonal by
Johnson–Lindenstrauss: the typical inner product between two of them is
`N^(-1/2)` (0.044 at `N=512`, measured 0.0442 at every density), so `V` enters
only logarithmically and a wider vocabulary is nearly free.

**What the freed budget buys.** With the dictionary gone, nearly every
trainable parameter is computation, and four mechanisms were measured onto that
budget — each on its own, at matched seed, zero-initialized at its output so
every arm starts from an identical loss:

- **Temporal attention** over the core's own state history. There are no layers
  to stack attention between, so it attends along *time*: each step queries a
  cache of earlier states. One KV head, RoPE at absolute stream position, and
  the cache crosses calls detached.
- **Delta-rule plasticity** — a `D×D` fast-weight matrix per stream, written by
  the sequence rather than trained. It writes the *error* (`v − Sk`) rather than
  the value, so a repeat write is a no-op and a stale association is erased
  instead of accumulated; `beta` is a full per-synapse gate field.
- **An auxiliary encoder and readout adapter** — a causal depthwise conv over
  injected codes, and a low-rank rotation into the codebook's frame so the
  state is not asked to be recurrent memory and a readout direction at once.
  Cheapest of the four and the most effective per parameter.
- **Muon** on every 2D operator, which costs no parameters at all and was the
  single largest win.

**What this notebook produces.** A trained model, a bits-per-byte score on
held-out text, the parameter report broken out by role, and the matched-seed
ablation arms that turn a number into a claim.
"""),

    md("## 1 — Environment\n\nPrints what we actually got, since Colab's "
       "assignment varies and every budget below depends on it."),
    code("""
import subprocess, sys, os, time, json, math, pathlib

print("--- GPU ---")
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
      or "no GPU — set Runtime > Change runtime type > GPU")

import torch
print(f"\\ntorch {torch.__version__}  cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"  {p.name}  {p.total_memory / 2**30:.1f} GiB  sm_{p.major}{p.minor}  {p.multi_processor_count} SMs")
    print(f"  bf16 supported: {torch.cuda.is_bf16_supported()}")
    # TF32 rather than autocast: the step applies the same few matrices T
    # times, so casting them per step costs 3*T extra kernels on a loop that
    # is already launch-bound — measured 31% slower than this, and less precise.
    torch.set_float32_matmul_precision('high')

IN_COLAB = 'google.colab' in sys.modules
print(f"\\ncolab: {IN_COLAB}")
"""),

    md("Clone and install. `datasets` and `tokenizers` are the only additions; "
       "torch is already present on Colab."),
    code(f"""
REPO = "{REPO}"
if IN_COLAB:
    if not pathlib.Path('aleph-repo').exists():
        # A local copy works too: upload the repo and skip this cell.
        r = subprocess.run(["git", "clone", "--depth", "1", REPO, "aleph-repo"],
                           capture_output=True, text=True)
        print(r.stdout or r.stderr)
    if pathlib.Path('aleph-repo').exists():
        os.chdir('aleph-repo')
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "datasets", "tokenizers"], check=False)

sys.path.insert(0, os.getcwd())
from aleph import Codebook, AlephCore, Trainer, __version__
print(f"aleph {{__version__}} from {{os.getcwd()}}")
"""),

    md("""Mount Drive so a checkpoint survives the session. Colab disconnects on
idle and on a 12/24-hour ceiling; a run that cannot resume loses everything it
had. Skip this cell to keep checkpoints session-local."""),
    code("""
CKPT_DIR = 'ckpt'
if IN_COLAB:
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        CKPT_DIR = '/content/drive/MyDrive/aleph_ckpt'
        os.makedirs(CKPT_DIR, exist_ok=True)
        print(f"checkpoints -> {CKPT_DIR}")
    except Exception as e:
        print(f"Drive unavailable ({type(e).__name__}); checkpoints stay session-local")
os.makedirs(CKPT_DIR, exist_ok=True)
"""),

    md("""## 2 — Corpus

TinyStories (Eldan & Li, [arXiv:2305.07759](https://arxiv.org/abs/2305.07759)):
stories restricted to the vocabulary and syntax of a young child, which is the
corpus that makes a few-million-parameter model worth measuring at all. On web
text the same model produces mush, and the score would describe the corpus's
difficulty rather than the architecture.

The BPE fit and the encode together take a few minutes and are cached, so
re-running this cell is free. `--stories` is the knob: 400k is ~130M tokens,
enough that a sub-1M-parameter model is not the thing that runs out."""),
    code("""
STORIES = 400_000     # ~130M tokens
VOCAB   = 4096        # nearly free under the codebook, so chosen for sequence length

t0 = time.perf_counter()
r = subprocess.run([sys.executable, "-u", "tools/build_corpus.py",
                    "--tokenizer", "bpe", "--vocab", str(VOCAB),
                    "--stories", str(STORIES)],
                   capture_output=True, text=True)
print(r.stdout[-2500:])
if r.returncode:
    print("STDERR:", r.stderr[-2000:])
print(f"\\n[{time.perf_counter() - t0:.0f}s]")
"""),

    code("""
from tools.experiment import find_cache
meta = find_cache('data')
print(f"corpus   {pathlib.Path(meta['path']).name}")
print(f"  train  {meta['train_tokens']:,} tokens")
print(f"  val    {meta['val_tokens']:,} tokens")
print(f"  V      {meta['vocab_size']}   {meta['bytes_per_token']:.2f} chars/token")
# bits/byte divides by the text the tokens encode, so this figure is part of
# every score below: a comparison against another cache is not a comparison.
"""),

    md("""## 3 — Smoke test

Asserts wiring, not learning. At this budget neither the loss nor the accuracy
moves measurably, so a gate on either would read the seed; what it checks
instead is deterministic — that the loss reaches every parameter family, that
the frozen code table does not move, and that `W`'s diagonal stays zero.

Run it before anything long. It fails in seconds where a real run fails in
hours."""),
    code("""
r = subprocess.run([sys.executable, "-u", "tools/experiment.py", "--mode", "smoke",
                    "--neurons", "128", "--depth", "4"],
                   capture_output=True, text=True)
print(r.stdout[-4000:])
if r.returncode:
    print("STDERR:", r.stderr[-3000:])
    raise SystemExit("smoke failed — fix before training")
"""),

    md("""## 4 — The run

Four facts measured on an RTX 3060 Ti set these defaults, and they are worth
knowing before changing them:

**The loop is launch-bound.** 128× the batch cost 1.15× the time — 1,042 tok/s
at batch 1 against 115,851 at batch 128. Batch is therefore nearly free and
should be as large as memory allows; it is the first thing to raise on a bigger
GPU.

**Wide and short beats narrow and long, for free.** The state and both branch
caches cross calls, so `seq` is a launch granularity rather than a context
length: a token still sees unbroken history and credit still reaches `window`
tokens back either way. At a fixed 16,384 tokens per step and identical VRAM
(1,660 MiB), `(batch 256, seq 64)` ran 3.6× faster than `(64, 256)` — 158,085
against 50,439 tok/s — *and* scored lower, because the step averages over 256
independent stream positions instead of 64. This is the single largest speedup
in the project and it costs nothing.

**`depth` saturates in time but not in capacity.** Branches are concatenated
into one wide matmul pair, so depth 4, 8 and 16 all cost ~151–158 ms while
parameters go 531k → 793k → 1.32M. Past depth 4 capacity is genuinely free,
which is why the default is 8.

**Warmup is not optional.** `depth=4` reached 0.0192 nats at lr 1e-3 and
diverged at 3e-3, where `depth=0` tolerated both. The extra gradient path
narrows the usable rate, so a depth comparison without warmup measures the
schedule."""),
    code("""
import torch
free = torch.cuda.get_device_properties(0).total_memory / 2**30 if torch.cuda.is_available() else 0

CFG = dict(
    neurons = 512,     # core is N^2; the matrix IS the model
    depth   = 8,       # free in time past 4, so take the capacity
    batch   = 512 if free > 30 else (256 if free > 14 else 64),
    seq     = 64,      # launch granularity, not context: state carries across
    window  = 32,      # how far credit travels back, independent of seq
    steps   = 6000,
    lr      = 1e-3,
    warmup  = 300,
)
print(f"{free:.0f} GiB -> batch {CFG['batch']}")
print(f"budget: {CFG['steps'] * CFG['batch'] * CFG['seq'] / 1e6:.0f}M tokens"
      f"  ({CFG['steps'] * CFG['batch'] * CFG['seq'] / meta['train_tokens']:.1f} epochs)")
# Epochs is the number that tells data starvation from over-parameterization:
# a corpus crossed dozens of times is starved, not too large a model.
"""),

    md("""The log line carries what distinguishes this architecture's failure
modes: `bpb` is the tokenizer-independent score, `ep` is epochs (starvation vs
over-parameterization), `|g|` and `clip%` are the gradient and how often the
clip binds, and `|h|` must sit at `√N` — 22.6 at `N=512` — because per-step
RMSNorm pins it there. A run that goes wrong goes wrong in one of those
columns."""),
    code("""
args = [sys.executable, "-u", "tools/experiment.py", "--mode", "train",
        "--neurons", str(CFG['neurons']), "--depth", str(CFG['depth']),
        "--batch", str(CFG['batch']), "--seq", str(CFG['seq']),
        "--window", str(CFG['window']),
        "--steps", str(CFG['steps']), "--lr", str(CFG['lr']),
        "--warmup", str(CFG['warmup']), "--ckpt", CKPT_DIR,
        "--tag", "main", "--sample", "--resume"]
print(" ".join(args), "\\n")

# Streamed rather than captured: a multi-hour cell that prints nothing is
# indistinguishable from one that has hung.
proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, bufsize=1)
for line in proc.stdout:
    print(line, end='')
proc.wait()
print(f"\\nexit {proc.returncode}")
"""),

    md("""## 5 — Ablations

A mechanism becomes a claim here, or does not. Each arm runs at the same seed
with one axis changed, and a negative result is reported as a negative result:
going looking for the configuration that rescues a mechanism is how an
efficiency claim stops being true.

`depth` first — the capacity that was free in wall-clock."""),
    code("""
r = subprocess.run([sys.executable, "-u", "tools/experiment.py", "--mode", "ablate",
                    "--ablate", "depth=0,4,8,16", "--steps", "1200",
                    "--neurons", str(CFG['neurons']), "--batch", str(CFG['batch']),
                    "--seq", str(CFG['seq']), "--lr", str(CFG['lr'])],
                   capture_output=True, text=True)
print(r.stdout[-3000:])
"""),

    md("""`nnz` — code density. The prediction is that it does not matter: the
typical inner product is `N^(-1/2)` at every density and only the worst-case
tail depends on `nnz`. If quality is flat here, the codebook's separability
argument holds."""),
    code("""
r = subprocess.run([sys.executable, "-u", "tools/experiment.py", "--mode", "ablate",
                    "--ablate", "nnz=8,32,64,256", "--steps", "1200",
                    "--neurons", str(CFG['neurons']), "--depth", str(CFG['depth']),
                    "--batch", str(CFG['batch']), "--seq", str(CFG['seq'])],
                   capture_output=True, text=True)
print(r.stdout[-3000:])
"""),

    md("""`neurons` — the parameter-efficiency curve itself. The core is `N²`, so
this is the axis a record claim is made along, and the interesting quantity is
bits per byte against trainable parameters rather than either alone."""),
    code("""
r = subprocess.run([sys.executable, "-u", "tools/experiment.py", "--mode", "ablate",
                    "--ablate", "neurons=128,256,512,768", "--steps", "1500",
                    "--depth", str(CFG['depth']), "--batch", str(CFG['batch']),
                    "--seq", str(CFG['seq'])],
                   capture_output=True, text=True)
print(r.stdout[-3000:])
"""),

    md("""### The synthesis arms

Each of the three branches was measured on its own before being turned on by
default, at 400 steps and 16,384 tokens per step from the same seed. All of
them are zero-initialized at their output projection, so every arm starts at an
identical loss and the difference is the mechanism rather than the
initialization:

| arm | params | val nats | Δ |
|---|---|---|---|
| AdamW baseline | 793,089 | 5.2949 | — |
| + Muon | 793,089 | 4.7643 | −0.531 |
| + aux encoder/readout | 861,185 | 4.5074 | −0.257 |
| + temporal attention | 957,505 | 4.6999 | −0.064 |
| + delta plasticity | 961,665 | 4.6452 | −0.119 |
| all four | 1,194,177 | 4.4094 | −0.885 |

Two things in that table are worth more than the totals. Muon is the largest
single win and it costs *zero* parameters — it is the same model, trained
differently, reaching in 400 steps what AdamW took 900 to reach. And the
cheapest mechanism is the most effective: the aux encoder and readout adapter
buy 0.257 nats for 68k parameters, because without them the state vector has to
serve as recurrent memory and as a readout direction in a frozen random frame
at the same time.

The branches then stack almost additively, which is the evidence that they are
doing different jobs rather than three versions of one.

Re-run it to confirm on your own hardware — each arm is about 90 seconds on a
T4, and the arms are independent so dropping one does not invalidate the
rest."""),
    code("""
r = subprocess.run([sys.executable, "-u", "tools/experiment.py", "--mode", "ablate",
                    "--ablate", "branches=none+adamw,none,aux,attn,plastic,attn+plastic+aux",
                    "--steps", "400", "--batch", "256", "--seq", "64"],
                   capture_output=True, text=True)
print(r.stdout[-3500:])
"""),

    md("""### Quantization

QAT on the recurrent operators, measured rather than assumed. The same matrix
is applied at every step, so a perturbation a feedforward layer would absorb
once gets re-applied `T` times here — which is why this is quantization-*aware*
training rather than a post-training pass:

| precision | val nats | Δ |
|---|---|---|
| fp32/tf32 | 4.4243 | — |
| int8 | 4.4383 | +0.014 |
| int4 | 4.6913 | +0.267 |
| ternary (1.58-bit) | 4.9016 | +0.477 |

int8 is free: 4× smaller weight storage for a difference inside run-to-run
noise. Below that the core pays real capacity, and the ternary figure lands on
the published expectation that 1.58-bit needs scale this model does not
have — so it is reported as a negative result rather than tuned until it
passes."""),
    code("""
r = subprocess.run([sys.executable, "-u", "tools/experiment.py", "--mode", "ablate",
                    "--ablate", "quant=none,int8,int4,ternary",
                    "--steps", "400", "--batch", "256", "--seq", "64"],
                   capture_output=True, text=True)
print(r.stdout[-3000:])
"""),

    md("""## 6 — Samples, and the honest accounting

Generation is the only check that reads as language rather than as a number.
A model can score well and produce nothing: read the samples."""),
    code("""
r = subprocess.run([sys.executable, "-u", "tools/experiment.py", "--mode", "eval",
                    "--ckpt", CKPT_DIR, "--tag", "main", "--sample"],
                   capture_output=True, text=True)
print(r.stdout[-4000:])
"""),

    md("""### What may and may not be claimed

The split matters more than the total, so the report prints four roles.

**Fair to say.** The *core* — the trained computation — is `N² − N` parameters,
and the token interface costs `V + N + 1` against the `2·V·N` a learned
embedding and head would cost. At `N=512, V=4096` that is 4,609 against
4,194,304. Quote bits per byte beside the trainable total and the cache it was
measured on.

**Not fair to say.** That the frozen table is free in the information-theoretic
sense — it is `V·nnz` fixed values, chosen by a seed rather than by training,
and a reader is entitled to count the seed's `V × N` support however they like.
State it as *trainable* parameters, which is what it is.

**Also not fair.** Comparing a bits-per-byte figure against one from a
different tokenizer or a rebuilt cache. The metric is tokenizer-independent but
it is defined against the held-out split it was computed on, so quote the
corpus's train/val token counts with any comparison — including between two of
our own runs."""),
    code("""
from aleph import Codebook, AlephCore
cb = Codebook(meta['vocab_size'], CFG['neurons'], device='cuda' if torch.cuda.is_available() else 'cpu')
m = AlephCore(CFG['neurons'], cb, depth=CFG['depth'])
r = m.param_report()
conv = 2 * meta['vocab_size'] * CFG['neurons']

print(f"{'role':<12} {'params':>12}")
for k in ('core', 'refine', 'interface', 'modulation'):
    print(f"{k:<12} {r[k]:>12,}")
print(f"{'TOTAL':<12} {r['total']:>12,}  trainable")
print(f"\\ninterface  {r['interface']:,} vs {conv:,} for a learned embed+head"
      f"   ({conv / r['interface']:.0f}x less)")
print(f"frozen     {cb.H.numel():,} table values, generated from seed {cb.seed:#x}")
print(f"           ({(cb.H != 0).sum().item():,} nonzero)")
"""),
]


def main():
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    out = pathlib.Path(__file__).resolve().parents[1] / 'notebooks' / 'aleph_colab.ipynb'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, indent=1), encoding='utf-8')
    n_code = sum(c['cell_type'] == 'code' for c in CELLS)
    print(f"{out}  ({len(CELLS)} cells: {n_code} code, {len(CELLS) - n_code} markdown)")


if __name__ == '__main__':
    main()
