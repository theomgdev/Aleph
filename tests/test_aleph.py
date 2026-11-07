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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from aleph.core.codebook import Codebook
from aleph.core.core import AlephCore
from aleph.data.stream import ByteTokenizer, TokenStream, build_cache
from aleph.train.trainer import Trainer

V, N = 512, 64


def make(depth=4, **kw):
    torch.manual_seed(0)
    return AlephCore(N, Codebook(V, N, nnz=16, **kw), depth=depth)


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
        assert {id(m.W), id(m.U), id(m.V)} == muon
        for p in (m.b, m.norm.weight, m.codebook.bias, m.codebook.in_gain):
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
