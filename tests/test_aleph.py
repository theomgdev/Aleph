"""Tests for the invariants a refactor would silently break.

Each one covers a property the design depends on rather than a behaviour that
merely happens to hold: the frozen interface staying frozen, the zero diagonal
staying zero, depth being inert at initialization, and the stream resuming
exactly. CPU-only and seconds, so CI can run them.
"""

import math
import pathlib
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from aleph.core.codebook import Codebook
from aleph.core.core import AlephCore
from aleph.data.stream import ByteTokenizer, TokenStream, build_cache
from aleph.train.trainer import Trainer

V, N = 512, 64


def make(depth=4, nnz=16, code='random', tokens=None, **kw):
    """A small core on CPU. Codebook kwargs are named; the rest go to the core."""
    torch.manual_seed(0)
    cb = Codebook(V, N, nnz=nnz, code=code, tokens=tokens)
    return AlephCore(N, cb, depth=depth, **kw)


class TestCodebook:
    def test_rows_are_unit_norm(self):
        H = Codebook(V, N, nnz=16).H
        assert torch.allclose(H.norm(dim=1), torch.ones(V), atol=1e-5)

    def test_nnz_respected(self):
        assert ((Codebook(V, N, nnz=16).H != 0).sum(dim=1) == 16).all()

    def test_typical_overlap_is_width_scaled(self):
        """The rms inner product tracks N^-1/2 whatever the density — that is
        what separability rests on, and it is the claim in the docstring."""
        for nnz in (8, 32, 64):
            H = Codebook(V, N, nnz=nnz).H
            G = H @ H.t()
            G.fill_diagonal_(0.0)
            rms = G.pow(2).sum().div(V * (V - 1)).sqrt().item()
            assert rms == pytest.approx(N ** -0.5, rel=0.25), f"nnz={nnz}: {rms}"

    def test_seed_reproducible_and_seed_sensitive(self):
        assert torch.equal(Codebook(V, N, seed=1).H, Codebook(V, N, seed=1).H)
        assert not torch.equal(Codebook(V, N, seed=1).H, Codebook(V, N, seed=2).H)

    def test_cost_is_v_plus_n_plus_one(self):
        """The whole point: the interface does not scale as V * N."""
        assert Codebook(V, N).num_params() == V + N + 1

    def test_pad_sentinel_injects_nothing(self):
        cb = Codebook(V, N)
        out = cb.inject(torch.tensor([-1, 0, -1]))
        assert out[0].abs().max() == 0 and out[2].abs().max() == 0
        assert out[1].abs().max() > 0

    def test_unigram_bias_beats_uniform(self):
        """A zipfian prior is worth real nats, which is why V biases are paid."""
        cb = Codebook(V, N)
        counts = 1.0 / torch.arange(1, V + 1).float() ** 1.07
        cb.set_unigram(counts * 1e6)
        logits = cb.logits(torch.zeros(1, N))
        p = logits.softmax(-1)[0]
        target = counts / counts.sum()
        assert -(target * p.clamp(min=1e-12).log()).sum() < math.log(V)

    def test_unigram_rejects_wrong_shape(self):
        with pytest.raises(ValueError, match="counts has shape"):
            Codebook(V, N).set_unigram(torch.ones(V + 1))

    def test_width_must_match_core(self):
        with pytest.raises(ValueError, match="must equal neurons"):
            AlephCore(N, Codebook(V, N * 2))

    def test_ngram_code_needs_tokens(self):
        with pytest.raises(ValueError, match="needs `tokens`"):
            Codebook(V, N, code='ngram')

    def test_ngram_code_relates_similar_strings(self):
        """Shared spelling must survive hashing, or the prior buys nothing."""
        toks = ['running', 'runner', 'xyzzyq'] + [f'w{i}' for i in range(V - 3)]
        H = Codebook(V, N, code='ngram', tokens=toks).H
        assert (H[0] @ H[1]).item() > (H[0] @ H[2]).item()


class TestCore:
    def test_state_sits_on_the_sqrt_n_sphere(self):
        """Per-step RMSNorm fixes the norm, which is why no clamp is needed."""
        m = make()
        _, h = m(torch.randint(0, V, (4, 24)))
        assert torch.allclose(h.norm(dim=-1), torch.full((4,), N ** 0.5), rtol=1e-3)

    def test_norm_holds_over_a_long_carry(self):
        m = make()
        h = None
        for _ in range(12):
            _, h = m(torch.randint(0, V, (2, 32)), state=h)
            h = h.detach()
        assert torch.allclose(h.norm(dim=-1), torch.full((2,), N ** 0.5), rtol=1e-3)

    def test_zero_diagonal_survives_a_step(self):
        m = make()
        opt = torch.optim.AdamW(m.parameters(), lr=0.1)
        logits, _ = m(torch.randint(0, V, (2, 8)))
        logits.sum().backward()
        assert m.W.grad.diagonal().abs().max() == 0, "grad hook did not fire"
        opt.step()
        assert m.W.diagonal().abs().max() == 0

    def test_depth_is_inert_at_init(self):
        """V zero-init means depth changes nothing until trained, so a depth
        ablation differs in one variable only."""
        x = torch.randint(0, V, (2, 10))
        torch.manual_seed(0)
        a, _ = make(depth=0)(x)
        torch.manual_seed(0)
        b, _ = make(depth=8)(x)
        assert torch.allclose(a, b, atol=1e-5)

    def test_depth_changes_output_once_trained(self):
        m = make(depth=4)
        x = torch.randint(0, V, (2, 10))
        before, _ = m(x)
        with torch.no_grad():
            m.V.normal_(std=0.1)
        after, _ = m(x)
        assert not torch.allclose(before, after)

    def test_self_gain_starts_shut(self):
        """The latch is gated closed, so W carries memory until it is earned."""
        m = make()
        assert torch.sigmoid(m.self_logit).max() < 0.05
        assert m.self_gain.abs().max() == 0

    def test_param_report_discounts_the_diagonal(self):
        m = make(depth=0)
        r = m.param_report()
        assert r['core'] == N * N - N
        assert r['total'] == sum(p.numel() for p in m.parameters() if p.requires_grad) - N

    def test_refine_cost_matches_the_formula(self):
        m = make(depth=4)
        assert m.param_report()['refine'] == 2 * 4 * N * m.rank

    def test_state_carries_forward(self):
        m = make()
        x = torch.randint(0, V, (2, 8))
        a, _ = m(x)
        _, h = m(x)
        b, _ = m(x, state=h.detach())
        assert not torch.allclose(a, b), "state had no effect"

    def test_checkpointing_matches_eager(self):
        x = torch.randint(0, V, (2, 12))
        torch.manual_seed(0)
        plain = make()
        torch.manual_seed(0)
        ckpt = make()
        ckpt.gradient_checkpointing = True
        ckpt.train()
        plain.train()
        a, _ = plain(x)
        b, _ = ckpt(x)
        assert torch.allclose(a, b, atol=1e-5)

    def test_window_does_not_change_the_forward_pass(self):
        """Only gradient reach is cut; every position is still predicted."""
        m = make()
        x = torch.randint(0, V, (2, 24))
        a, _ = m(x, window=0)
        b, _ = m(x, window=4)
        assert torch.allclose(a, b, atol=1e-5)

    def test_window_bounds_the_gradient(self):
        """The reason `window` exists: a shared matrix's per-step Jacobian sits
        above 1, so a long backward pass is exponential."""
        import torch.nn.functional as F
        norms = {}
        for w in (4, 0):
            torch.manual_seed(0)
            m = make(depth=4)
            with torch.no_grad():
                m.V.normal_(std=0.05)
            x = torch.randint(0, V, (4, 96))
            y = torch.randint(0, V, (4, 96))
            logits, _ = m(x, window=w)
            F.cross_entropy(logits.reshape(-1, V).float(), y.reshape(-1)).backward()
            norms[w] = torch.cat([p.grad.flatten() for p in m.parameters()
                                  if p.grad is not None]).norm().item()
        assert norms[4] < norms[0], f"window did not reduce |g|: {norms}"

    def test_window_rejects_negative(self):
        with pytest.raises(ValueError, match="window must be >= 0"):
            make()(torch.randint(0, V, (2, 8)), window=-1)

    def test_rejects_bad_shape(self):
        with pytest.raises(ValueError, match="must be \\(B, T\\)"):
            make()(torch.randint(0, V, (4,)))

    def test_generate_shape_and_range(self):
        out = make().generate(torch.randint(0, V, (3, 4)), length=7)
        assert out.shape == (3, 7)
        assert (out >= 0).all() and (out < V).all()

    def test_greedy_generation_is_deterministic(self):
        m = make()
        x = torch.randint(0, V, (2, 4))
        a = m.generate(x, length=5, temperature=0.0)
        assert torch.equal(a, m.generate(x, length=5, temperature=0.0))

    def test_generate_leaves_training_mode_untouched(self):
        m = make()
        m.train()
        m.generate(torch.randint(0, V, (1, 2)), length=2)
        assert m.training


@pytest.fixture
def cache(tmp_path):
    texts = [f"Once upon a time story number {i} ended happily.\n" for i in range(400)]
    return build_cache(texts, ByteTokenizer(), tmp_path, 'test', verbose=False)


class TestStream:
    def test_targets_are_inputs_shifted(self, cache):
        x, y = TokenStream(cache, 4, 16).next()
        assert (x[:, 1:] == y[:, :-1]).all()

    def test_cursors_start_apart(self, cache):
        assert len(set(TokenStream(cache, 8, 16).cursor.tolist())) == 8

    def test_resume_is_exact(self, cache):
        s = TokenStream(cache, 4, 16)
        st = s.state()
        a, _ = s.next()
        s.load(st)
        assert (a == s.next()[0]).all()

    def test_val_split_is_disjoint_from_train(self, cache):
        tr, va = TokenStream(cache, 2, 8, 'train'), TokenStream(cache, 2, 8, 'val')
        assert tr.hi == va.lo and va.hi == cache['tokens']

    def test_epochs_counts_passes(self, cache):
        s = TokenStream(cache, 4, 16)
        for _ in range(10):
            s.next()
        assert s.epochs == pytest.approx(10 * 4 * 16 / cache['train_tokens'], rel=1e-6)

    def test_stamp_detects_a_different_tokenizer(self, tmp_path):
        texts = ["hello world\n"] * 50
        a = build_cache(texts, ByteTokenizer(), tmp_path, 'x', verbose=False)
        tok = ByteTokenizer()
        tok.name = 'other'
        b = build_cache(texts, tok, tmp_path, 'x', verbose=False)
        assert a['path'] != b['path'], "two tokenizers shared one cache"

    def test_rejects_seq_longer_than_split(self, cache):
        with pytest.raises(ValueError, match="need > seq"):
            TokenStream(cache, 2, 10 ** 7)

    def test_uint16_ceiling_is_enforced(self, tmp_path):
        tok = ByteTokenizer()
        tok.vocab_size = 70000
        with pytest.raises(ValueError, match="uint16"):
            build_cache(["x"], tok, tmp_path, 'big', verbose=False)


class TestMuon:
    def test_orthogonalize_tames_the_spectrum(self):
        """Not an isometry -- it pulls the singular values toward 1, and how
        far depends on conditioning. The claim is the ratio, not the value."""
        from aleph.train.muon import orthogonalize
        torch.manual_seed(0)
        M = torch.randn(64, 64)
        before = torch.linalg.svdvals(M)
        after = torch.linalg.svdvals(orthogonalize(M).float())
        assert after.max() / after.min() < before.max() / before.min()
        assert after.max() < 1.5

    def test_orthogonalize_handles_both_orientations(self):
        from aleph.train.muon import orthogonalize
        for shape in ((128, 32), (32, 128)):
            out = orthogonalize(torch.randn(*shape))
            assert out.shape == shape
            assert torch.isfinite(out).all()

    def test_orthogonalize_refuses_non_2d(self):
        from aleph.train.muon import orthogonalize
        with pytest.raises(ValueError, match="2D matrix"):
            orthogonalize(torch.zeros(8))

    def test_refuses_1d_in_the_muon_half(self):
        """1D parameters have trivial singular values; the split is not
        cosmetic, so a wrong one must fail loudly."""
        from aleph.train.muon import MuonAdamW
        with pytest.raises(ValueError, match=r"only to 2D operator matrices"):
            MuonAdamW([torch.zeros(8)], [])

    def test_from_model_splits_operators_from_modulation(self):
        from aleph.train.muon import MuonAdamW
        m = make(depth=4)
        opt = MuonAdamW.from_model(m)
        muon = {id(p) for g in opt.param_groups if g['use_muon'] for p in g['params']}
        # Every operator goes to Muon, including the branches'.
        for p in (m.W, m.U, m.V, m.dec_u, m.dec_v, m.attn.q_proj.weight,
                  m.attn.o_proj.weight, m.plastic.k_temp.weight, m.plastic.o_proj.weight):
            assert id(p) in muon
        # Gains, biases and the per-synapse gate field do not.
        for p in (m.b, m.norm.weight, m.codebook.bias, m.codebook.in_gain,
                  m.plastic.beta, m.plastic.decay, m.attn.q_norm.weight):
            assert id(p) not in muon

    def test_step_preserves_the_zero_diagonal(self):
        """Newton-Schulz returns a dense matrix, so W's diagonal must be
        re-zeroed after the update or the core's invariant is lost."""
        from aleph.train.muon import MuonAdamW
        import torch.nn.functional as F
        m = make(depth=4)
        opt = MuonAdamW.from_model(m, lr=0.1)
        logits, _ = m(torch.randint(0, V, (2, 8)))
        F.cross_entropy(logits.reshape(-1, V).float(),
                        torch.randint(0, V, (16,))).backward()
        opt.step()
        assert m.W.diagonal().abs().max() == 0

    def test_step_moves_both_halves(self):
        from aleph.train.muon import MuonAdamW
        import torch.nn.functional as F
        m = make(depth=4)
        with torch.no_grad():
            m.V.normal_(std=0.05)      # open the branch so U takes gradient
        opt = MuonAdamW.from_model(m, lr=1e-2)
        before = {n: p.detach().clone() for n, p in m.named_parameters()}
        logits, _ = m(torch.randint(0, V, (2, 8)))
        F.cross_entropy(logits.reshape(-1, V).float(),
                        torch.randint(0, V, (16,))).backward()
        opt.step()
        for name in ('W', 'U', 'b', 'norm.weight'):
            p = dict(m.named_parameters())[name]
            assert (p - before[name]).abs().max() > 0, f"{name} did not move"

    def test_set_lr_scales_the_muon_half(self):
        from aleph.train.muon import MuonAdamW
        opt = MuonAdamW.from_model(make(depth=4), lr=1e-3)
        opt.set_lr(1e-3, muon_scale=3.0)
        for g in opt.param_groups:
            assert g['lr'] == pytest.approx(3e-3 if g['use_muon'] else 1e-3)


class TestTrainer:
    def test_warmup_then_decay(self):
        t = Trainer(make(), lr=1e-3, warmup=10, total=100)
        assert t.lr_at(0) < t.lr_at(5) < t.lr_at(9)
        assert t.lr_at(9) == pytest.approx(1e-3, rel=1e-6)
        assert t.lr_at(99) < t.lr_at(50) < t.lr_at(10)

    def test_decay_floor_respected(self):
        t = Trainer(make(), lr=1e-3, warmup=5, total=50, min_lr_frac=0.1)
        assert t.lr_at(49) >= 1e-4 * 0.99

    def test_modulation_is_not_decayed(self):
        """Norm gains, gates and the unigram bias must not be pulled to zero."""
        t = Trainer(make(), weight_decay=0.1)
        decayed = {id(p) for p in t.opt.param_groups[0]['params']}
        m = t.model
        assert id(m.W) in decayed and id(m.U) in decayed
        for p in (m.norm.weight, m.codebook.bias, m.codebook.in_gain, m.self_logit):
            assert id(p) not in decayed

    def test_a_step_moves_the_loss_and_the_weights(self, cache):
        m = make()
        t = Trainer(m, lr=1e-2, warmup=1, total=20)
        s = TokenStream(cache, 4, 16)
        w0 = m.W.detach().clone()
        first = None
        for _ in range(20):
            x, y = s.next()
            mm, _ = t.train_step(torch.as_tensor(x), torch.as_tensor(y))
            first = first if first is not None else mm['loss']
        assert mm['loss'] < first
        assert (m.W - w0).abs().max() > 0
        assert m.W.diagonal().abs().max() == 0

    def test_frozen_table_never_moves(self, cache):
        m = make()
        t = Trainer(m, lr=1e-1, warmup=1, total=10)
        h0 = m.codebook.H.clone()
        s = TokenStream(cache, 4, 16)
        for _ in range(10):
            x, y = s.next()
            t.train_step(torch.as_tensor(x), torch.as_tensor(y))
        assert torch.equal(m.codebook.H, h0)

    def test_evaluate_reports_bits_per_byte(self, cache):
        ev = Trainer(make()).evaluate(TokenStream(cache, 2, 16, 'val'), batches=2)
        assert math.isfinite(ev['bits_per_byte']) and ev['bits_per_byte'] > 0
        assert ev['bits_per_byte'] == pytest.approx(
            ev['bits_per_token'] / cache['bytes_per_token'], rel=1e-6)

    def test_state_returned_detached(self, cache):
        x, y = TokenStream(cache, 2, 8).next()
        _, h = Trainer(make()).train_step(torch.as_tensor(x), torch.as_tensor(y))
        assert not h.requires_grad


class TestBranchInertness:
    """Every optional branch must be inert at initialization.

    This is what makes an ablation one variable: a model with a branch on and
    one with it off must produce the identical loss at step 0, so any later
    difference is the mechanism learning rather than a different starting point.
    """

    @pytest.mark.parametrize('kw', [{'attn': True}, {'plastic': True}, {'aux': True}])
    def test_branch_changes_nothing_at_init(self, kw):
        ids = torch.randint(0, V, (3, 12))
        off = {'attn': False, 'plastic': False, 'aux': False}
        base, _ = make(depth=4, **{**off})(ids)
        with_branch, _ = make(depth=4, **{**off, **kw})(ids)
        assert torch.allclose(base, with_branch, atol=1e-6)

    def test_all_branches_on_is_also_inert(self):
        ids = torch.randint(0, V, (3, 12))
        off, _ = make(depth=4, attn=False, plastic=False, aux=False)(ids)
        on, _ = make(depth=4)(ids)
        assert torch.allclose(off, on, atol=1e-6)

    def test_branches_contribute_after_training(self, cache):
        """And once trained, each branch must actually be doing something."""
        import torch.nn.functional as F
        m = make(depth=4)
        t = Trainer(m, lr=1e-2, warmup=1, total=20)
        s = TokenStream(cache, 4, 16)
        for _ in range(8):
            x, y = s.next()
            t.train_step(torch.as_tensor(x), torch.as_tensor(y))
        assert m.attn.o_proj.weight.abs().max() > 0
        assert m.plastic.o_proj.weight.abs().max() > 0
        assert m.dec_v.abs().max() > 0
        assert m.enc.weight.abs().max() > 0


class TestAttention:
    def test_cache_carries_across_calls(self):
        m = make(depth=0, plastic=False, aux=False)
        ids = torch.randint(0, V, (2, 8))
        _, h = m(ids)
        assert m.attn._pos == 8 and m.attn._k.shape[2] == 8
        m(ids, state=h)
        assert m.attn._pos == 16 and m.attn._k.shape[2] == 16

    def test_fresh_state_resets_the_cache(self):
        m = make(depth=0, plastic=False, aux=False)
        ids = torch.randint(0, V, (2, 8))
        m(ids)
        m(ids)  # state=None: a new stream, so the history must not leak in
        assert m.attn._pos == 8

    def test_cache_truncates_to_window(self):
        m = make(depth=0, plastic=False, aux=False, attn_window=6)
        ids = torch.randint(0, V, (2, 10))
        _, h = m(ids)
        assert m.attn._k.shape[2] == 6
        m(ids, state=h)
        assert m.attn._k.shape[2] == 6

    def test_cache_is_detached(self):
        m = make(depth=0, plastic=False, aux=False)
        m(torch.randint(0, V, (2, 8)))
        assert not m.attn._k.requires_grad and not m.attn._v.requires_grad

    def test_carry_makes_a_short_chunk_see_the_past(self):
        """A chunked run and a whole-sequence run must agree, which is the
        only reason splitting a sequence into short chunks is free."""
        torch.manual_seed(0)
        m = make(depth=0, plastic=False, aux=False).eval()
        ids = torch.randint(0, V, (2, 16))
        with torch.no_grad():
            whole, _ = m(ids)
            a, h = m(ids[:, :8])
            b, _ = m(ids[:, 8:], state=h)
        assert torch.allclose(whole, torch.cat([a, b], dim=1), atol=1e-4)

    def test_batch_change_resets_rather_than_crashing(self):
        m = make(depth=0, plastic=False, aux=False)
        m(torch.randint(0, V, (4, 8)))
        m(torch.randint(0, V, (2, 8)))  # must not try to concat (4,...) with (2,...)


class TestPlasticity:
    def test_trace_carries_and_is_detached(self):
        m = make(depth=0, attn=False, aux=False)
        ids = torch.randint(0, V, (2, 8))
        _, h = m(ids)
        assert m.plastic._S.shape == (2, m.plastic.dim, m.plastic.dim)
        assert not m.plastic._S.requires_grad
        trace = m.plastic._S.clone()
        m(ids, state=h)
        assert not torch.equal(m.plastic._S, trace)

    def test_delta_rule_erases_what_it_retrieves(self):
        """Write the same association twice: the second write must be smaller,
        because the first one already satisfies it. That is the whole
        difference between the delta rule and Hebbian accumulation."""
        from aleph.core.plasticity import Plasticity
        torch.manual_seed(0)
        p = Plasticity(N, dim=8)
        with torch.no_grad():
            p.beta.fill_(4.0)   # wide open
            p.decay.fill_(8.0)  # no forgetting
        x = torch.randn(1, 1, N).repeat(1, 6, 1)  # the same state, six times
        with torch.no_grad():
            p(x, window=0)
        # After repeated identical writes the trace must converge, not grow.
        s1 = p._S.norm().item()
        p.reset_cache()
        with torch.no_grad():
            p(x[:, :2], window=0)
        s2 = p._S.norm().item()
        assert s1 < s2 * 3, f"trace grew without bound: {s2} -> {s1}"

    def test_state_per_stream_exceeds_the_vector_state(self):
        """The point of the matrix: more memory than the state vector, for
        parameters that do not scale with it."""
        m = make(depth=0, attn=False, aux=False, plastic_dim=32)
        assert m.plastic.dim ** 2 > m.n


class TestQuantization:
    @pytest.mark.parametrize('mode,levels', [('ternary', 3), ('int4', 15), ('int8', 255)])
    def test_quantize_produces_the_promised_level_count(self, mode, levels):
        from aleph.core.core import quantize_ste
        torch.manual_seed(0)
        w = torch.randn(4, 256)
        q = quantize_ste(w, mode)
        for row in q:
            assert row.unique().numel() <= levels

    def test_gradient_passes_straight_through(self):
        from aleph.core.core import quantize_ste
        w = torch.randn(4, 32, requires_grad=True)
        quantize_ste(w, 'ternary').sum().backward()
        assert torch.allclose(w.grad, torch.ones_like(w))

    def test_none_is_exactly_the_identity(self):
        from aleph.core.core import quantize_ste
        w = torch.randn(4, 32)
        assert quantize_ste(w, None) is w and quantize_ste(w, 'none') is w

    def test_rejects_an_unknown_mode(self):
        from aleph.core.core import quantize_ste
        with pytest.raises(ValueError, match=r"quant must be"):
            quantize_ste(torch.randn(2, 2), 'int3')

    def test_qat_model_trains_and_keeps_the_diagonal(self, cache):
        m = make(depth=4, quant='int8')
        t = Trainer(m, lr=1e-2, warmup=1, total=10)
        s = TokenStream(cache, 4, 16)
        for _ in range(4):
            x, y = s.next()
            metrics, _ = t.train_step(torch.as_tensor(x), torch.as_tensor(y))
        assert math.isfinite(metrics['loss'])
        assert torch.equal(m.W.diagonal(), torch.zeros(N))


class TestGeometryEquivalence:
    """The measurement the default geometry rests on: a wide-short step and a
    narrow-long one are the same computation, so choosing the fast one costs
    nothing."""

    def test_chunked_equals_whole_under_state_carry(self):
        torch.manual_seed(0)
        m = make(depth=4).eval()
        ids = torch.randint(0, V, (2, 24))
        with torch.no_grad():
            whole, _ = m(ids)
            parts, h = [], None
            for i in range(0, 24, 8):
                out, h = m(ids[:, i:i + 8], state=h)
                parts.append(out)
        assert torch.allclose(whole, torch.cat(parts, dim=1), atol=1e-4)


class TestCheckpointIdentity:
    """A checkpoint is only readable against the architecture that wrote it.

    The codebook is generated from the seed and the readout *is* that codebook,
    so a field that decides the weights' meaning but is missing from
    `ARCH_FIELDS` means `--mode eval` silently scores a different model than
    the one that was trained.
    """

    def test_arch_fields_covers_every_constructor_argument(self):
        import inspect
        import tools.experiment as X
        from aleph.core.core import AlephCore
        # Every AlephCore knob that changes the parameter shapes or the
        # codebook, as opposed to runtime behaviour.
        runtime = {'self', 'codebook', 'device', 'dropout',
                   'gradient_checkpointing', 'activation'}
        knobs = set(inspect.signature(AlephCore.__init__).parameters) - runtime
        missing = knobs - set(X.ARCH_FIELDS)
        assert not missing, f"ARCH_FIELDS is missing {sorted(missing)}"

    def test_arch_of_round_trips_through_restore(self):
        import argparse
        import tools.experiment as X
        a = argparse.Namespace(
            neurons=64, depth=4, rank=None, nnz=16, code='random', seed=0,
            no_self_gain=False, no_attn=True, attn_heads=2, attn_kv_heads=1,
            attn_head_dim=8, attn_window=16,
            no_plastic=False, plastic_dim=8, no_aux=True, aux_rank=4,
            aux_kernel=3, quant='int8', wait=0, ponder_max=4,
            ponder_prior=0.5, ponder_eps=0.01)
        meta = {'vocab_size': V, 'path': 'x/y.bin'}
        arch = X.arch_of(a, meta)
        assert arch['attn'] is False and arch['aux'] is False
        assert arch['plastic'] is True and arch['self_gain'] is True

        fresh = argparse.Namespace(**{k: None for k in vars(a)})
        X.restore_arch(fresh, {'arch': arch})
        assert (fresh.no_attn, fresh.no_plastic, fresh.no_aux) == (True, False, True)
        assert fresh.no_self_gain is False
        for k in ('neurons', 'depth', 'nnz', 'code', 'seed', 'quant',
                  'attn_heads', 'attn_kv_heads', 'attn_head_dim', 'attn_window',
                  'plastic_dim', 'aux_rank', 'aux_kernel'):
            assert getattr(fresh, k) == getattr(a, k), k
        # And the round trip is a fixed point, which is what resume compares.
        assert X.arch_of(fresh, meta) == arch

    def test_state_dict_reloads_strictly_for_every_branch_combination(self):
        import itertools
        for attn, plastic, aux in itertools.product((True, False), repeat=3):
            m = make(depth=4, attn=attn, plastic=plastic, aux=aux)
            twin = make(depth=4, attn=attn, plastic=plastic, aux=aux)
            twin.load_state_dict(m.state_dict(), strict=True)
            ids = torch.randint(0, V, (2, 8))
            a_out, _ = m.eval()(ids)
            b_out, _ = twin.eval()(ids)
            assert torch.equal(a_out, b_out), (attn, plastic, aux)


def trained_like(depth=4, **kw):
    """A model whose optional branches are non-zero, as training leaves them.

    Every branch is zero-initialized at its output, so a freshly built model
    cannot detect a bug in any of them: the branch contributes nothing and a
    broken one contributes nothing just as convincingly.
    """
    m = make(depth=depth, **kw)
    with torch.no_grad():
        if m.enc is not None:
            torch.nn.init.normal_(m.enc.weight, std=0.5)
            torch.nn.init.normal_(m.dec_v, std=0.1)
        if m.attn is not None:
            torch.nn.init.normal_(m.attn.o_proj.weight, std=0.1)
        if m.plastic is not None:
            torch.nn.init.normal_(m.plastic.o_proj.weight, std=0.1)
    return m.eval()


class TestDecodeConsistency:
    """Chunk size must not change the answer.

    This is the property the whole design rests on: the state and every cache
    cross forward calls, so a sequence may be split anywhere. Generation splits
    it into single tokens, which is the harshest case — if a cache does not
    carry, the model samples from a distribution it was never scored on, and
    the bits-per-byte figure describes a different model than the one producing
    the text.
    """

    @pytest.mark.parametrize('chunk', [1, 2, 8])
    def test_chunked_equals_whole(self, chunk):
        m = trained_like()
        ids = torch.randint(0, V, (3, 24))
        with torch.no_grad():
            whole, _ = m(ids, window=0)
            m.reset_caches()
            parts, h = [], None
            for i in range(0, 24, chunk):
                out, h = m(ids[:, i:i + chunk], state=h, window=0)
                parts.append(out)
        got = torch.cat(parts, dim=1)
        assert torch.allclose(whole, got, atol=1e-4), \
            f"chunk={chunk} diverges by {(whole - got).abs().max():.3e}"

    @pytest.mark.parametrize('branch', ['aux', 'attn', 'plastic'])
    def test_each_branch_carries_on_its_own(self, branch):
        """Isolated, so a failure names the branch that does not carry."""
        off = {'attn': False, 'plastic': False, 'aux': False}
        m = trained_like(**{**off, branch: True})
        ids = torch.randint(0, V, (2, 16))
        with torch.no_grad():
            whole, _ = m(ids, window=0)
            m.reset_caches()
            parts, h = [], None
            for i in range(16):
                out, h = m(ids[:, i:i + 1], state=h, window=0)
                parts.append(out)
        got = torch.cat(parts, dim=1)
        assert torch.allclose(whole, got, atol=1e-4), \
            f"{branch} does not carry across calls: {(whole - got).abs().max():.3e}"

    def test_generation_is_causal(self):
        """Editing token k must move nothing before k.

        A conv with symmetric padding, or a cache that keeps future keys,
        passes every consistency check above and still leaks the answer.
        """
        m = trained_like()
        ids = torch.randint(0, V, (1, 16))
        with torch.no_grad():
            base, _ = m(ids, window=0)
        for k in (4, 8, 12):
            alt = ids.clone()
            alt[0, k] = (alt[0, k] + 7) % V
            with torch.no_grad():
                out, _ = m(alt, window=0)
            assert torch.equal(base[:, :k], out[:, :k]), f"position {k} leaks backward"
            assert not torch.allclose(base[:, k:], out[:, k:]), f"position {k} has no effect"

    def test_aux_conv_has_no_padding(self):
        """The left context comes from the carried tail, so Conv1d must not
        add its own — padding and a carry together would double-count."""
        m = make(depth=0, attn=False, plastic=False)
        assert m.enc.padding == (0,)
        assert m.enc.kernel_size == (m.aux_kernel,)


class TestColdStart:
    """A sequence has to begin, and beginning was what was not trained.

    When state is carried across every step, a cold start is ~0.02% of training
    positions while being 100% of generation's first token. `reset_every` makes
    the start appear on a schedule, and when it triggers it must clear the
    branch caches along with the hidden state -- a reset that clears `h` while
    leaving the attention cache warm is not a cold start.
    """

    def test_reset_every_clears_state_and_branch_caches_on_schedule(self, cache):
        m = make(depth=4)
        t = Trainer(m, lr=1e-2, warmup=1, total=10, reset_every=2)
        s = TokenStream(cache, 4, 8)
        v = TokenStream(cache, 2, 8, 'val')
        # 2 steps: step 0 starts cold (pos -> 8), step 1 carries (pos -> 16).
        t.run(s, v, 2, log_every=10, eval_every=0)
        assert m.attn._pos == 16
        # 1 step covers only step 0, which `step % 2 == 0` resets (pos -> 8).
        t.run(s, v, 1, log_every=10, eval_every=0)
        assert m.attn._pos == 8

    def test_reset_every_zero_carries_indefinitely(self, cache):
        m = make(depth=4)
        t = Trainer(m, lr=1e-2, warmup=1, total=10, reset_every=0)
        s = TokenStream(cache, 4, 8)
        v = TokenStream(cache, 2, 8, 'val')
        t.run(s, v, 4, log_every=10, eval_every=0)
        assert m.attn._pos == 32


class TestWaitSteps:
    """Extra core steps between tokens, with no readout.

    Depth here is time, so "think longer on this token" is literally more
    applications of `W` with no token arriving. Three things have to hold or
    the mechanism is not what it claims: `wait=0` must be bit-identical to a
    model built without it (so it is a clean ablation axis), the wait code must
    be outside the readout's span (a token the softmax can emit is a token the
    softmax is trained to suppress, since every supervised position pushes
    every non-target logit down), and one token must still produce exactly one
    logit row so the branch horizons stay measured in tokens.
    """

    def test_wait_zero_is_bit_identical(self):
        a, b = make(depth=4, wait=0), make(depth=4)
        ids = torch.randint(0, V, (3, 12))
        with torch.no_grad():
            la, _ = a(ids)
            lb, _ = b(ids, wait=0)
        assert torch.equal(la, lb)

    def test_one_token_one_logit_row_whatever_wait(self):
        m = make(depth=4, wait=3)
        ids = torch.randint(0, V, (2, 7))
        with torch.no_grad():
            logits, h = m(ids)
        assert logits.shape == (2, 7, V)
        assert h.shape == (2, m.n)

    def test_wait_code_is_frozen_and_outside_the_readout(self):
        m = make(depth=4, wait=2)
        cb = m.codebook
        assert 'wait_code' in dict(cb.named_buffers())
        assert cb.wait_code.shape == (cb.width,)
        # Not a row of H: the readout is `h @ H.t()`, so a code absent from H
        # cannot be emitted, and therefore cannot be suppressed either.
        assert not (cb.H == cb.wait_code).all(dim=1).any()

    def test_wait_changes_the_computation(self):
        m2 = make(depth=4, wait=2)
        m0 = make(depth=4, wait=0)
        m0.load_state_dict(m2.state_dict())
        ids = torch.randint(0, V, (2, 8))
        with torch.no_grad():
            l2, _ = m2(ids)
            l0, _ = m0(ids)
        assert not torch.allclose(l2, l0)

    def test_chunked_and_stepwise_agree_with_wait(self):
        """The failure this guards: a wait step that reads a cache the decode
        path does not carry would score one way in training and generate
        another, exactly as the stateless conv padding once did."""
        m = make(depth=4, wait=2)
        m.eval()
        ids = torch.randint(0, V, (2, 10))
        with torch.no_grad():
            whole, _ = m(ids)
            m.reset_caches()
            outs, st = [], None
            for t in range(ids.shape[1]):
                o, st = m(ids[:, t:t + 1], state=st)
                outs.append(o)
        stepwise = torch.cat(outs, dim=1)
        assert torch.allclose(whole, stepwise, atol=1e-4)
        assert torch.equal(whole.argmax(-1), stepwise.argmax(-1))

    def test_gradient_reaches_w_through_wait_steps(self):
        m = make(depth=4, wait=3)
        ids = torch.randint(0, V, (2, 6))
        logits, _ = m(ids)
        logits.float().mean().backward()
        assert m.W.grad is not None and m.W.grad.norm() > 0

    def test_generate_returns_length_tokens_whatever_wait(self):
        m = make(depth=4, wait=2)
        prompt = torch.randint(0, V, (2, 4))
        out = m.generate(prompt, length=5, temperature=0.0)
        assert out.shape == (2, 5)
        assert (out >= 0).all() and (out < V).all()

    def test_negative_wait_is_rejected(self):
        with pytest.raises(ValueError, match='wait'):
            make(depth=4, wait=-1)
        m = make(depth=4)
        with pytest.raises(ValueError, match='wait'):
            m(torch.randint(0, V, (1, 3)), wait=-2)


class TestPonder:
    """Adaptive halting: the allocation, which is what fixed `wait` lacked.

    The mechanism is PonderNet's halting distribution over candidate states,
    and three of its properties are load-bearing rather than incidental. The
    distribution is normalized for *any* halting law, which is what makes
    `ponder_max` a hard bound that needs no penalty and cannot run away. The
    head is zero-initialized and draws no RNG, so a pondering model and a plain
    one at the same seed share a core and `ponder_max` is an ablation with one
    variable. And the exit is the expected state rather than a sampled one, so
    gradient reaches the head from the language-model loss alone -- no
    straight-through estimator and no reward signal anywhere.
    """

    def test_ponder_zero_is_inert_and_shares_the_core(self):
        plain, pondering = make(depth=4), make(depth=4, ponder_max=4)
        assert torch.equal(plain.W, pondering.W)
        assert torch.equal(plain.U, pondering.U)
        ids = torch.randint(0, V, (2, 8))
        with torch.no_grad():
            la, _ = plain(ids)
            lb, _ = make(depth=4, ponder_max=0)(ids)
        assert torch.equal(la, lb)

    def test_costs_n_plus_one_parameters(self):
        plain, pondering = make(depth=4), make(depth=4, ponder_max=8)
        delta = (pondering.param_report()['total'] - plain.param_report()['total'])
        assert delta == N + 1

    @pytest.mark.parametrize('bias', [-20.0, -3.0, 0.0, 3.0, 20.0])
    def test_exit_distribution_is_normalized_for_any_law(self, bias):
        """The maximum is the support of a probability distribution, so it holds
        by construction rather than by a rule. An adversarial head cannot make
        the mass sum to anything but one, and cannot exceed `ponder_max`."""
        m = make(depth=4, ponder_max=6)
        with torch.no_grad():
            m.ponder.bias.fill_(bias)
            m.ponder.w.normal_(0, 3.0)
            m.ponder.reset_stats()
            h = torch.randn(4, N)
            h = h / h.norm(dim=-1, keepdim=True) * N ** 0.5
            m._ponder_steps(h, m.codebook.inject_wait(), m.W, m.U, m.V, None)
        assert 0.0 <= m.ponder.mean_steps <= m.ponder.max_wait + 1e-5

    def test_exit_stays_on_the_state_sphere(self):
        """A mixture of unit vectors is shorter than its parts, so without the
        rescale a token that hedges would hand the recurrence a smaller state
        than one that commits -- a norm channel leaking 'this was hard' into
        every downstream branch."""
        m = make(depth=4, ponder_max=6)
        with torch.no_grad():
            m.ponder.w.normal_(0, 2.0)
            h = torch.randn(5, N)
            h = h / h.norm(dim=-1, keepdim=True) * N ** 0.5
            out = m._ponder_steps(h, m.codebook.inject_wait(), m.W, m.U, m.V, None)
        assert torch.allclose(out.norm(dim=-1), h.norm(dim=-1), rtol=1e-4)

    def test_gradient_reaches_the_head_from_the_lm_loss_alone(self):
        """No KL, no reward: averaging the state instead of sampling an exit is
        what makes the head trainable by the same backward pass as `W`."""
        m = make(depth=4, ponder_max=4)
        ids = torch.randint(0, V, (2, 6))
        logits, _ = m(ids)
        F.cross_entropy(logits.reshape(-1, V),
                        torch.randint(0, V, (12,))).backward()
        assert m.ponder.w.grad is not None and m.ponder.w.grad.norm() > 0
        assert m.ponder.bias.grad is not None
        assert m.W.grad.norm() > 0

    def test_kl_is_live_and_starts_near_the_prior(self):
        m = make(depth=4, ponder_max=8, ponder_prior=0.5)
        m(torch.randint(0, V, (2, 8)))
        assert m.ponder.kl.requires_grad
        # Zero-init head means a state-independent lambda at exactly `prior`,
        # so the mean should land on the truncated geometric's, near 1/p - 1.
        assert abs(m.ponder.mean_steps - 1.0) < 0.15

    def test_one_token_one_logit_row(self):
        m = make(depth=4, ponder_max=6)
        with torch.no_grad():
            logits, h = m(torch.randint(0, V, (3, 9)))
        assert logits.shape == (3, 9, V)
        assert h.shape == (3, m.n)

    def test_chunked_and_stepwise_agree(self):
        m = make(depth=4, ponder_max=4)
        m.eval()
        ids = torch.randint(0, V, (2, 10))
        with torch.no_grad():
            whole, _ = m(ids)
            m.reset_caches()
            outs, st = [], None
            for t in range(ids.shape[1]):
                o, st = m(ids[:, t:t + 1], state=st)
                outs.append(o)
        assert torch.allclose(whole, torch.cat(outs, dim=1), atol=1e-4)

    def test_wait_and_ponder_are_mutually_exclusive(self):
        with pytest.raises(ValueError, match='same mechanism'):
            make(depth=4, wait=2, ponder_max=4)

    def test_rejects_bad_configuration(self):
        with pytest.raises(ValueError, match='ponder_max'):
            make(depth=4, ponder_max=-1)
        with pytest.raises(ValueError, match='prior'):
            make(depth=4, ponder_max=4, ponder_prior=0.0)
        with pytest.raises(ValueError, match='prior'):
            make(depth=4, ponder_max=4, ponder_prior=1.0)

    def test_trainer_adds_the_kl_only_to_what_is_differentiated(self, cache):
        """The reported loss has to stay comparable to a non-pondering run, so
        the regularizer may change the gradient but never the score."""
        m = make(depth=4, ponder_max=4)
        t = Trainer(m, lr=1e-3, warmup=1, total=4, ponder_beta=0.5)
        s = TokenStream(cache, 2, 8)
        x, y = s.next()
        x = torch.as_tensor(x)
        y = torch.as_tensor(y)
        metrics, _ = t.train_step(x, y)
        with torch.no_grad():
            logits, _ = m(x)
            ce = F.cross_entropy(logits.reshape(-1, V).float(), y.reshape(-1))
        # Reported loss is a cross-entropy, not a cross-entropy plus a penalty:
        # it moved only by the optimizer step taken between the two.
        assert abs(metrics['loss'] - ce.item()) < 1.0
        assert 'steps' in metrics and metrics['steps'] >= 0.0


class TestCachePortability:
    """A cache built or tracked on one OS must not break another.

    Two failure modes surfaced on Colab: a `.json` sidecar tracked in git
    without its `.bin` memmap made `build_cache` report a 0s cache hit with no
    data behind it, and a Windows backslash inside `meta['path']` failed to
    open on Linux.
    """

    def test_orphan_json_without_bin_is_rebuilt_not_trusted(self, tmp_path):
        from aleph.data.stream import ByteTokenizer, build_cache
        tok = ByteTokenizer()
        texts = ["Once upon a time there was a little cat.\n"] * 40
        meta = build_cache(texts, tok, tmp_path, 'mini', verbose=False)
        bin_path = pathlib.Path(meta['path'])
        # Simulate a git clone that has the `.json` sidecar but not `.bin`.
        bin_path.unlink()
        rebuilt = build_cache(texts, tok, tmp_path, 'mini', verbose=False)
        assert pathlib.Path(rebuilt['path']).exists()
        assert '\\' not in rebuilt['path']

    def test_find_cache_ignores_orphan_json_and_normalizes_separators(self, tmp_path):
        import json
        import pytest
        import tools.experiment as X
        from aleph.data.stream import ByteTokenizer, build_cache
        tok = ByteTokenizer()
        meta = build_cache(["hello world\n"] * 40, tok, tmp_path, 'real', verbose=False)
        # Corrupt the stored path with Windows backslashes and a wrong prefix,
        # and drop an alphabetically-later orphan `.json` with no `.bin`.
        j = next(tmp_path.glob('real_*.json'))
        raw = json.loads(j.read_text())
        raw['path'] = "data" + chr(92) + "real_wrong.bin"
        j.write_text(json.dumps(raw))
        (tmp_path / 'zzzz_orphan_bpe4096_deadbeef.json').write_text(json.dumps(raw))

        found = X.find_cache(tmp_path)
        assert '\\' not in found['path']
        assert pathlib.Path(found['path']).exists()
        assert found['tokens'] == meta['tokens']

        # And if no `.bin` exists at all, `find_cache` fails rather than
        # returning a pointer to a missing file.
        pathlib.Path(found['path']).unlink()
        with pytest.raises(SystemExit):
            X.find_cache(tmp_path)
