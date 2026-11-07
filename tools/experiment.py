"""Train Aleph on TinyStories and report what it cost.

Modes:
  smoke   — every path in seconds, asserting wiring rather than learning
  train   — a real run, checkpointed and resumable
  ablate  — matched-seed arms over one axis
  eval    — re-score a saved checkpoint, no GPU-minutes

The parameter report is printed as four roles rather than one total, because a
parameter-efficiency claim lives or dies on which of them a reader counts: the
`core` is the model, `interface` is what a conventional design would pay
`2 * V * N` for, and quoting a single number hides the trade the architecture
exists to make.
"""

import argparse
import json
import math
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from aleph.core.codebook import Codebook  # noqa: E402
from aleph.core.core import AlephCore  # noqa: E402
from aleph.data.stream import TokenStream, unigram_counts  # noqa: E402
from aleph.train.trainer import Trainer  # noqa: E402

# The knobs a user actually turns mid-investigation, in one block.
NEURONS = 512
DEPTH = 8
# Wide and short. Every token still sees an unbroken state and every gradient
# still reaches 32 tokens back -- the carry across calls makes `seq` a launch
# granularity, not a context length. At a fixed 16,384 tokens per step,
# (256, 64) measured 3.6x faster than (64, 256) *and* scored lower, because
# the step averages over 256 independent stream positions instead of 64.
BATCH = 256
SEQ = 64
STEPS = 4000
LR = 1e-3
WARMUP = 200

# Architecture identity: a checkpoint is only readable against the same values,
# since the codebook is generated from them and the readout is that codebook.
# Everything that decides what the weights mean. The resume guard compares
# these and `--mode eval` restores from them, so a field missing here is a
# checkpoint that silently loads into the wrong architecture.
ARCH_FIELDS = ('neurons', 'depth', 'rank', 'vocab', 'nnz', 'code', 'seed', 'cache',
               'self_gain', 'attn', 'attn_heads', 'attn_kv_heads',
               'attn_head_dim', 'attn_window', 'plastic', 'plastic_dim',
               'aux', 'aux_rank', 'aux_kernel', 'quant')


def find_cache(data_dir, name=None):
    metas = sorted(pathlib.Path(data_dir).glob('*.json'))
    metas = [m for m in metas
             if 'bpe_' not in m.name and m.with_suffix('.bin').exists()]
    if not metas:
        raise SystemExit(f"no cache in {data_dir}/ — run tools/build_corpus.py first")
    if name:
        metas = [m for m in metas if name in m.name] or metas
    chosen = metas[-1]
    meta = json.loads(chosen.read_text())
    meta['path'] = chosen.with_suffix('.bin').as_posix()
    return meta


def build(a, meta, verbose=True):
    torch.manual_seed(a.seed)
    dev = torch.device(a.device)
    tokens = None
    if a.code == 'ngram':
        bpe = pathlib.Path(a.data) / f"bpe_{meta['vocab_size']}.json"
        if not bpe.exists():
            raise SystemExit(f"code='ngram' needs token strings: {bpe} missing")
        from aleph.data.stream import BPETokenizer
        tok = BPETokenizer(meta['vocab_size'])
        tok.train(iter(()), bpe, verbose=False)
        tokens = tok.tokens

    cb = Codebook(meta['vocab_size'], a.neurons, nnz=a.nnz, code=a.code,
                  tokens=tokens, seed=a.seed, device=dev)
    if not a.no_unigram:
        cb.set_unigram(unigram_counts(meta, meta['vocab_size']))
    model = AlephCore(a.neurons, cb, depth=a.depth, rank=a.rank,
                      activation=a.activation, dropout=a.dropout,
                      self_gain=not a.no_self_gain,
                      attn=not a.no_attn, attn_heads=a.attn_heads,
                      attn_kv_heads=a.attn_kv_heads,
                      attn_head_dim=a.attn_head_dim, attn_window=a.attn_window,
                      plastic=not a.no_plastic, plastic_dim=a.plastic_dim,
                      aux=not a.no_aux, aux_rank=a.aux_rank,
                      aux_kernel=a.aux_kernel, quant=a.quant,
                      gradient_checkpointing=a.checkpoint, device=dev).to(dev)

    if verbose:
        r = model.param_report()
        conv = 2 * meta['vocab_size'] * a.neurons
        branches = [n for n, on in (('attn', model.attn is not None),
                                    ('plastic', model.plastic is not None),
                                    ('aux', model.enc is not None)) if on]
        print(f"  N={a.neurons} depth={a.depth} rank={model.rank} V={meta['vocab_size']}")
        print(f"  branches: {', '.join(branches) or 'none'}"
              f"{'  quant=' + a.quant if a.quant else ''}")
        print(f"  trainable {r['total']:,}  = core {r['core']:,} + refine {r['refine']:,}"
              f" + interface {r['interface']:,} + mod {r['modulation']:,}")
        print(f"  interface vs a learned embed+head: {r['interface']:,} against "
              f"{conv:,}  ({conv / max(r['interface'], 1):.0f}x less)")
        print(f"  code coherence {cb.coherence():.4f} (worst-case overlap; "
              f"rms is N^-1/2 = {a.neurons ** -0.5:.4f})")
    return model


def arch_of(a, meta):
    return {'neurons': a.neurons, 'depth': a.depth, 'rank': a.rank,
            'vocab': meta['vocab_size'], 'nnz': a.nnz, 'code': a.code,
            'seed': a.seed, 'cache': pathlib.Path(meta['path']).name,
            'self_gain': not a.no_self_gain,
            'attn': not a.no_attn, 'attn_heads': a.attn_heads,
            'attn_kv_heads': a.attn_kv_heads,
            'attn_head_dim': a.attn_head_dim, 'attn_window': a.attn_window,
            'plastic': not a.no_plastic, 'plastic_dim': a.plastic_dim,
            'aux': not a.no_aux, 'aux_rank': a.aux_rank,
            'aux_kernel': a.aux_kernel, 'quant': a.quant}


def streams(a, meta):
    return (TokenStream(meta, a.batch, a.seq, 'train', seed=a.seed),
            TokenStream(meta, min(a.batch, 32), a.seq, 'val', seed=0))


def run_train(a, meta):
    model = build(a, meta)
    tr, va = streams(a, meta)
    trainer = Trainer(model, lr=a.lr, warmup=a.warmup, total=a.steps,
                      weight_decay=a.weight_decay, window=a.window,
                      optimizer=a.optimizer, reset_every=a.reset_every,
                      device=torch.device(a.device))

    ck = pathlib.Path(a.ckpt) / f"aleph_{a.tag}.pt"
    ck.parent.mkdir(parents=True, exist_ok=True)
    if ck.exists() and not (a.resume or a.overwrite):
        raise SystemExit(f"{ck} exists — pass --resume to continue or --overwrite to replace")

    start = 0
    if a.resume and ck.exists():
        # weights_only=True: the payload is tensors, dicts and scalars only, so
        # nothing here needs arbitrary unpickling.
        blob = torch.load(ck, map_location=a.device, weights_only=True)
        saved = {k: blob['arch'][k] for k in ARCH_FIELDS}
        if saved != arch_of(a, meta):
            raise SystemExit(f"checkpoint architecture differs:\n  saved {saved}\n  now   {arch_of(a, meta)}")
        model.load_state_dict(blob['model'])
        trainer.opt.load_state_dict(blob['opt'])
        trainer.step_count = start = blob['step']
        tr.load(blob['stream'])
        print(f"  resumed at step {start:,}, val loss {blob['best']:.4f}")

    best = [float('inf')]

    def on_eval(step, ev, improved):
        if not improved:
            return
        best[0] = ev['loss']
        torch.save({'model': model.state_dict(), 'opt': trainer.opt.state_dict(),
                    'step': trainer.step_count, 'best': ev['loss'],
                    'arch': arch_of(a, meta), 'stream': tr.state(),
                    'metrics': ev}, ck)

    print(f"\ntraining {a.steps - start:,} steps "
          f"({(a.steps - start) * a.batch * a.seq / 1e6:.1f}M tokens)\n")
    out = trainer.run(tr, va, a.steps - start, log_every=a.log_every,
                      eval_every=a.eval_every, on_eval=on_eval)

    ev = trainer.evaluate(va, 40)
    r = model.param_report()
    print(f"\n{'=' * 64}")
    print(f"  final    val loss {ev['loss']:.4f}  bpb {ev['bits_per_byte']:.4f}  "
          f"ppl {ev['perplexity']:.2f}")
    print(f"  params   {r['total']:,} trainable ({r['core']:,} core)")
    print(f"  data     {out['tokens'] / 1e6:.1f}M tokens, {out['epochs']:.2f} epochs, "
          f"{out['tokens'] / out['seconds']:,.0f} tok/s")
    print(f"  bpb per 1k params: {ev['bits_per_byte'] / (r['total'] / 1000):.5f}")
    print(f"{'=' * 64}")
    if a.sample:
        show_samples(model, meta, a)
    return ev


def show_samples(model, meta, a):
    """Generation is the only check that reads as language rather than a number."""
    from aleph.data.stream import BPETokenizer, ByteTokenizer
    if meta['tokenizer'] == 'byte':
        tok = ByteTokenizer()
    else:
        tok = BPETokenizer(meta['vocab_size'])
        tok.train(iter(()), pathlib.Path(a.data) / f"bpe_{meta['vocab_size']}.json",
                  verbose=False)
    print("\n--- samples ---")
    for prompt in ("Once upon a time", "The little girl", "Tom and Sara"):
        ids = torch.tensor([tok.encode(prompt)], device=a.device)
        # 0.6, not the conventional 0.8: measured mean predictive entropy here
        # is 3.27 nats at top-1 probability 0.390, so 0.8 draws from the tail of
        # a distribution that is genuinely uncertain, and the text degrades
        # while the score does not.
        out = model.generate(ids, length=70, temperature=a.temperature, top_k=40)
        print(f"  {prompt!r} -> {tok.decode(out[0].tolist())!r}")


def run_ablate(a, meta):
    """Matched-seed arms over one axis, reported including when negative."""
    axis, values = a.ablate.split('=')
    values = [int(v) if v.isdigit() else v for v in values.split(',')]
    print(f"ablating {axis} over {values} at seed {a.seed}, {a.steps} steps each\n")
    rows = []
    for v in values:
        if axis == 'branches':
            # The synthesis axis: each arm names what it keeps, joined with
            # '+', so the comparison is between architectures rather than
            # between scalars. 'none' means no branches.
            keep = {k for k in v.split('+') if k != 'none'}
            unknown = keep - {'attn', 'plastic', 'aux', 'adamw'}
            if unknown:
                raise SystemExit(f"branches: unknown {sorted(unknown)}; "
                                 "pick from attn, plastic, aux, adamw")
            a.no_attn, a.no_plastic = 'attn' not in keep, 'plastic' not in keep
            a.no_aux = 'aux' not in keep
            a.optimizer = 'adamw' if 'adamw' in keep else 'muon'
        else:
            setattr(a, axis, None if (axis == 'quant' and v == 'none') else v)
        print(f"--- {axis}={v} ---")
        model = build(a, meta, verbose=False)
        tr, va = streams(a, meta)
        trainer = Trainer(model, lr=a.lr, warmup=a.warmup, total=a.steps,
                          weight_decay=a.weight_decay, window=a.window,
                          optimizer=a.optimizer, reset_every=a.reset_every,
                      device=torch.device(a.device))
        t0 = time.perf_counter()
        trainer.run(tr, va, a.steps, log_every=max(a.steps // 4, 1), eval_every=0)
        ev = trainer.evaluate(va, 30)
        r = model.param_report()
        rows.append((v, r['total'], ev['loss'], ev['bits_per_byte'],
                     time.perf_counter() - t0))
        print(f"  val loss {ev['loss']:.4f}  bpb {ev['bits_per_byte']:.4f}  "
              f"{r['total']:,} params\n")

    print(f"{'=' * 72}")
    width = max(10, max(len(str(v)) for v, *_ in rows))
    print(f"{axis:>{width}} {'params':>10} {'val loss':>10} {'bpb':>8} {'vs first':>10} {'sec':>7}")
    for v, p, loss, bpb, secs in rows:
        print(f"{str(v):>{width}} {p:>10,} {loss:>10.4f} {bpb:>8.4f} "
              f"{loss - rows[0][2]:>+10.4f} {secs:>7.0f}")
    print(f"{'=' * 72}")
    return rows


def run_smoke(a, meta):
    """Assert wiring, in seconds. Never asserts a loss or an accuracy: at this
    budget neither moves measurably, so a gate on either reads the seed."""
    a.steps, a.batch, a.seq = 12, 4, 32
    print("--- smoke ---")
    model = build(a, meta)
    tr, va = streams(a, meta)
    trainer = Trainer(model, lr=a.lr, warmup=2, total=12, window=a.window,
                      optimizer=a.optimizer, reset_every=a.reset_every,
                      device=torch.device(a.device))

    print("\n  trainable parameters:")
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(f"    {n:<28} {tuple(p.shape)}")
    frozen = {'codebook.H'}
    for n, b in model.named_buffers():
        if n in frozen:
            print(f"    {n:<28} {tuple(b.shape)}  FROZEN")

    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    h0 = model.codebook.H.clone()

    x, y = tr.next()
    x = torch.as_tensor(x, device=a.device)
    y = torch.as_tensor(y, device=a.device)
    logits, _ = model(x)
    assert logits.shape == (a.batch, a.seq, meta['vocab_size']), logits.shape

    # Every family expected to learn must be reachable from the loss. This is
    # deterministic and catches the detached-graph and wrong-group bugs a smoke
    # exists for, where a loss threshold would only read the seed.
    #
    # Three families are held closed BY DESIGN at step 0 and are excluded:
    # `V` is zero-initialized, so `act(norm(out) @ U) @ V` is identically zero
    # and neither `U` nor `r_norm` can take gradient until `V` moves; `self_gain`
    # is zero, so `self_logit` sits behind a zero factor for the same reason.
    # Each takes gradient on its own schedule once the tensor in front of it
    # moves -- which is the mechanism working, not a detached graph.
    import torch.nn.functional as F
    loss = F.cross_entropy(logits.reshape(-1, meta['vocab_size']).float(), y.reshape(-1))
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, [p for _, p in named], allow_unused=True)
    # Everything behind a zero-initialized output projection. Their gradient is
    # exactly zero at step 0 by construction -- that zero is what makes each
    # branch inert at init, so an ablation differs in one variable. They open
    # once the tensor in front of them moves, which the next block checks.
    gated = ({'self_logit', 'U', 'r_norm.weight', 'dec_u', 'dec_norm.weight'}
             | {n for n, _ in named if n.startswith(('attn.', 'plastic.'))
                and not n.endswith('o_proj.weight')})
    print("\n  gradient reaches:")
    for (n, _), g in zip(named, grads):
        norm = 0.0 if g is None else g.norm().item()
        note = ''
        if n in gated:
            note = '   (behind a zero-init projection, by design)'
        elif norm == 0:
            note = '   <-- UNREACHED'
        print(f"    {n:<28} |g| {norm:.3e}{note}")
    unreached = [n for (n, _), g in zip(named, grads)
                 if (g is None or g.norm().item() == 0) and n not in gated]
    assert not unreached, f"loss does not reach {unreached}"

    # The gated families must open once what precedes them has moved, so a few
    # steps of training is what actually tests them.
    for _ in range(6):
        xx, yy = tr.next()
        trainer.train_step(torch.as_tensor(xx, device=a.device),
                           torch.as_tensor(yy, device=a.device))
    logits, _ = model(x)
    loss = F.cross_entropy(logits.reshape(-1, meta['vocab_size']).float(), y.reshape(-1))
    grads = dict(zip([n for n, _ in named],
                     torch.autograd.grad(loss, [p for _, p in named], allow_unused=True)))
    print("\n  after 6 steps, the gated families:")
    for n in sorted(gated):
        g = grads.get(n)
        norm = 0.0 if g is None else g.norm().item()
        print(f"    {n:<28} |g| {norm:.3e}{'   <-- STILL SHUT' if norm == 0 else '   open'}")
    assert (grads['U'] is not None and grads['U'].norm() > 0), \
        "U never opened — the refinement branches are unreachable, not merely gated"

    out = trainer.run(tr, va, a.steps, log_every=6, eval_every=12, eval_batches=2)
    moved = [n for n, p in model.named_parameters()
             if p.requires_grad and (p - before[n]).abs().max() > 0]
    print(f"\n  {len(moved)}/{len(named)} trainable tensors moved in {a.steps} steps")
    assert torch.equal(model.codebook.H, h0), "frozen code table moved"
    assert model.W.diagonal().abs().max() == 0, "zero diagonal violated"
    assert math.isfinite(out['best_val_loss']), "val loss not finite"

    g = model.generate(x[:, :4], length=6)
    assert g.shape == (a.batch, 6), g.shape
    print(f"  generate {tuple(g.shape)} ok | code table frozen | W diag 0 | val finite")
    print("\n  SMOKE PASS")


def restore_arch(a, blob):
    """Put a checkpoint's architecture back onto the args before building.

    `arch_of` writes the booleans the way the model reads them (`attn: True`),
    while the CLI carries the negations (`--no-attn`), so the two have to be
    translated rather than copied. A field handled here but absent from an older
    checkpoint keeps whatever the CLI default is, which is the only safe
    fallback: it is visible in the `branches:` line that `build` prints.
    """
    arch = blob['arch']
    for k, v in arch.items():
        if k in ('vocab', 'cache'):
            continue  # identity of the corpus, not a constructor argument
        if k in ('attn', 'plastic', 'aux', 'self_gain'):
            setattr(a, f'no_{k}', not v)
        else:
            setattr(a, k, v)
    return a


def run_eval(a, meta):
    ck = pathlib.Path(a.ckpt) / f"aleph_{a.tag}.pt"
    if not ck.exists():
        raise SystemExit(f"{ck} not found")
    blob = torch.load(ck, map_location=a.device, weights_only=True)
    restore_arch(a, blob)
    model = build(a, meta)
    # strict: the branches were just restored from the checkpoint's own arch, so
    # a missing or unexpected key means that restoration is wrong and the score
    # would be measured on a different model than the one that was trained.
    model.load_state_dict(blob['model'], strict=True)
    _, va = streams(a, meta)
    trainer = Trainer(model, device=torch.device(a.device))
    ev = trainer.evaluate(va, 60)
    print(f"\n  step {blob['step']:,}  val loss {ev['loss']:.4f}  "
          f"bpb {ev['bits_per_byte']:.4f}  ppl {ev['perplexity']:.2f}")
    if a.sample:
        show_samples(model, meta, a)
    return ev


def run_depends(a, meta):
    """What has the trained model come to depend on?

    An ablation measures what a mechanism adds when the rest of the model can
    train around it. This measures what the finished weights cannot do without,
    by intervening on a trained checkpoint and re-scoring. The two are different
    quantities and the second is usually much larger.

    For the stateful branches it also separates memory from capacity: shortening
    how long a cache may persist says how far back the model is actually
    reading, as opposed to how far it was allowed to.
    """
    ck = pathlib.Path(a.ckpt) / f"aleph_{a.tag}.pt"
    if not ck.exists():
        raise SystemExit(f"{ck} not found")
    blob = torch.load(ck, map_location=a.device, weights_only=True)
    restore_arch(a, blob)
    model = build(a, meta, verbose=False)
    model.load_state_dict(blob['model'], strict=True)
    _, va = streams(a, meta)
    trainer = Trainer(model, device=torch.device(a.device))

    def score():
        model.reset_caches()
        ev = trainer.evaluate(va, 40)
        return ev['loss'], ev['bits_per_byte']

    base = score()
    print(f"\n  step {blob['step']:,}  {base[0]:.4f} nats  {base[1]:.4f} bpb\n")
    print(f"  {'intervention':<40} {'nats':>8} {'bpb':>8} {'delta':>9}")
    print(f"  {'-' * 68}")

    def row(label, res):
        print(f"  {label:<40} {res[0]:>8.4f} {res[1]:>8.4f} {res[0] - base[0]:>+9.4f}")

    row('(none)', base)

    for name, branch in (('attention', model.attn), ('plasticity', model.plastic)):
        if branch is None:
            continue
        saved = branch.forward
        branch.forward = lambda *args, **kw: torch.zeros_like(args[0])
        row(f'{name} branch zeroed', score())
        branch.forward = saved

    if model.attn is not None:
        full = model.attn.window
        for w in (1, 8, 32, 64):
            if w >= full:
                continue
            model.attn.window = w
            row(f'attention cache limited to {w}', score())
        model.attn.window = full

    if model.plastic is not None:
        # The trace reads before it writes, so resetting it every token is
        # exactly deleting the branch. Limiting the horizon is the real question.
        saved = model.plastic.forward
        for k in (2, 16, 64):
            def chunked(h_seq, window=32, _k=k):
                out = []
                for i in range(0, h_seq.shape[1], _k):
                    model.plastic.reset_cache()
                    out.append(saved(h_seq[:, i:i + _k], window=0))
                return torch.cat(out, dim=1)
            model.plastic.forward = chunked
            row(f'plasticity trace limited to {k}', score())
        model.plastic.forward = saved

    for label, tensor in (('aux encoder (conv)', getattr(model.enc, 'weight', None)),
                          ('aux readout adapter', model.dec_v),
                          ('unigram logit bias', model.codebook.bias)):
        if tensor is None:
            continue
        saved = tensor.detach().clone()
        with torch.no_grad():
            tensor.zero_()
        row(f'{label} zeroed', score())
        with torch.no_grad():
            tensor.copy_(saved)

    # How much of the readout is context-dependent at all? The bias is one
    # fixed vector added at every position, so comparing its spread against the
    # state-dependent term says how much of a logit the core actually decided.
    with torch.no_grad():
        x, _ = va.next()
        logits, _ = model(torch.as_tensor(x, device=a.device), window=0)
        signal = (logits - model.codebook.bias).std(-1).mean().item()
        bias = model.codebook.bias.std().item()
    print(f"  {'-' * 68}")
    print(f"\n  readout decomposition: state-dependent std {signal:.4f}, "
          f"fixed bias std {bias:.4f}")
    print(f"  signal/bias = {signal / max(bias, 1e-9):.3f}"
          "   (below 1 means most of a logit is a corpus statistic)")
    print("\n  A branch whose removal costs nothing was not doing anything. A cache\n"
          "  whose horizon stops mattering past k is reading k tokens back, not\n"
          "  the window it was given.")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', choices=('train', 'smoke', 'ablate', 'eval', 'depends'),
                   default='train',
                   help="depends: intervene on a trained checkpoint to see what "
                        "the finished weights rely on, which is not what an "
                        "ablation measures.")

    g = p.add_argument_group('architecture')
    g.add_argument('--neurons', type=int, default=NEURONS)
    g.add_argument('--depth', type=int, default=DEPTH)
    g.add_argument('--rank', type=int, default=None)
    g.add_argument('--nnz', type=int, default=64)
    g.add_argument('--code', choices=('random', 'ngram'), default='random')
    g.add_argument('--activation', default='gelu')
    g.add_argument('--no-unigram', action='store_true',
                   help="Leave the logit bias at zero — the arm that measures "
                        "what the unigram prior was worth.")
    g.add_argument('--no-self-gain', action='store_true',
                   help="Drop the per-neuron self-connection latch.")
    g.add_argument('--no-attn', action='store_true', help="Drop temporal attention.")
    g.add_argument('--attn-heads', type=int, default=4)
    g.add_argument('--attn-kv-heads', type=int, default=1,
                   help="Shared KV heads. The cache, not the projections, is the cost.")
    g.add_argument('--attn-head-dim', type=int, default=16)
    g.add_argument('--attn-window', type=int, default=128)
    g.add_argument('--no-plastic', action='store_true', help="Drop delta-rule fast weights.")
    g.add_argument('--plastic-dim', type=int, default=32)
    g.add_argument('--no-aux', action='store_true', help="Drop the encoder and readout adapter.")
    g.add_argument('--aux-rank', type=int, default=64)
    g.add_argument('--aux-kernel', type=int, default=4,
                   help="Width of the causal conv over injected codes: the n in n-gram.")
    g.add_argument('--quant', choices=('int8', 'int4', 'ternary'), default=None,
                   help="QAT on the recurrent operators. int8 measured free; "
                        "int4 costs +0.27 nats, ternary +0.48.")

    g = p.add_argument_group('training')
    g.add_argument('--steps', type=int, default=STEPS)
    g.add_argument('--batch', type=int, default=BATCH)
    g.add_argument('--seq', type=int, default=SEQ)
    g.add_argument('--lr', type=float, default=LR)
    g.add_argument('--warmup', type=int, default=WARMUP)
    g.add_argument('--window', type=int, default=32,
                   help="Backward window in steps. The lever that makes a long "
                        "sequence trainable: a shared matrix's per-step Jacobian "
                        "sits above 1, so 0 (full backprop) is exponential.")
    g.add_argument('--optimizer', choices=('muon', 'adamw'), default='muon',
                   help="Muon measured -0.53 val nats against AdamW at equal "
                        "parameters, seed and step count.")
    g.add_argument('--reset-every', type=int, default=16,
                   help="Begin a fresh sequence every N steps so cold starts "
                        "get gradient. 0 carries state forever, which measured "
                        "11.54 nats at position 0 against 2.74 warm.")
    g.add_argument('--weight-decay', type=float, default=0.01)
    g.add_argument('--dropout', type=float, default=0.0)
    g.add_argument('--checkpoint', action='store_true',
                   help="Gradient checkpointing: step-count-independent memory, ~1.7x slower.")

    g = p.add_argument_group('run')
    g.add_argument('--data', default='data')
    g.add_argument('--cache', default=None)
    g.add_argument('--ckpt', default='ckpt')
    g.add_argument('--tag', default='main')
    g.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    g.add_argument('--seed', type=int, default=0)
    g.add_argument('--resume', action='store_true')
    g.add_argument('--overwrite', action='store_true')
    g.add_argument('--log-every', type=int, default=50)
    g.add_argument('--eval-every', type=int, default=500)
    g.add_argument('--sample', action='store_true')
    g.add_argument('--temperature', type=float, default=0.6,
                   help="Sampling temperature for --sample. 0.6 rather than 0.8: "
                        "this model's predictive entropy is 3.27 nats, so 0.8 "
                        "samples the tail.")
    g.add_argument('--ablate', default=None, metavar='AXIS=V1,V2',
                   help="e.g. --ablate depth=0,4,8, or --ablate "
                        "branches=none+adamw,none,aux,attn,plastic,attn+plastic+aux "
                        "to compare architectures (names join with '+')")

    a = p.parse_args()
    if a.mode == 'ablate' and not a.ablate:
        p.error("--mode ablate needs --ablate AXIS=V1,V2")
    return a


def main():
    a = parse_args()
    meta = find_cache(a.data, a.cache)
    print(f"corpus {pathlib.Path(meta['path']).name}: "
          f"{meta['train_tokens']:,} train / {meta['val_tokens']:,} val tokens, "
          f"V={meta['vocab_size']}, {meta['bytes_per_token']:.2f} chars/token")
    {'train': run_train, 'smoke': run_smoke, 'ablate': run_ablate,
     'eval': run_eval, 'depends': run_depends}[a.mode](a, meta)


if __name__ == '__main__':
    main()
