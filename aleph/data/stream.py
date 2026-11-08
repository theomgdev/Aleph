"""Corpus to token stream, with the tokenizer sized against the core.

Two jobs, and the second is the one that matters for a parameter claim.

**Tokenization.** A byte vocabulary needs no table but inflates the sequence
~5.4x against words, and sequence length is depth in a recurrent model. A BPE
vocabulary trained in-domain compresses that back down. Since the codebook
makes vocabulary nearly free -- `V` logit biases rather than `V x N` of table --
the usual reason to keep `V` small is gone, and `V` can be chosen for sequence
length alone.

**Streaming.** The cache is a flat `uint16` memmap and a batch is `B` windows
into it. Windows are drawn from `B` independent cursors spread evenly across the
corpus, each advancing by `T` and wrapping independently, so a batch never
contains two windows from the same neighbourhood and the state carried between
calls stays continuous per row. The cursor vector is the whole stream state, so
a resumed run reads exactly where it stopped.

The cache is stamped with the corpus identity, the tokenizer and `V`: a score
is only meaningful against the split it was measured on, and a silently reused
cache from a different tokenizer would move the held-out boundary.
"""

import hashlib
import json
import pathlib

import numpy as np


def _stamp(**kw):
    return hashlib.blake2b(json.dumps(kw, sort_keys=True).encode(), digest_size=16).hexdigest()


class ByteTokenizer:
    """Bytes as tokens. No table, no training, 5.4x the sequence length."""

    name = 'byte'

    def __init__(self):
        self.vocab_size = 256
        self.tokens = [repr(bytes([i]))[2:-1] for i in range(256)]

    def encode(self, text):
        return list(text.encode('utf-8', 'replace'))

    def encode_batch(self, texts):
        return [self.encode(t) for t in texts]

    def decode(self, ids):
        return bytes(i for i in ids if 0 <= i < 256).decode('utf-8', 'replace')


class BPETokenizer:
    """Byte-level BPE trained on the corpus itself.

    In-domain and small: the point is to cut sequence length, and a vocabulary
    trained on the text that will be modelled does that at a fraction of the
    size a general-purpose vocabulary needs.
    """

    name = 'bpe'

    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self._tok = None
        self.tokens = None

    def train(self, text_iter, path, verbose=True):
        path = pathlib.Path(path)
        try:
            from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
        except ImportError as exc:
            raise ImportError(
                "BPE needs `tokenizers` (pip install tokenizers); "
                "ByteTokenizer needs nothing and is the fallback") from exc

        if path.exists():
            self._tok = Tokenizer.from_file(str(path))
            if self._tok.get_vocab_size() != self.vocab_size:
                if verbose:
                    print(f"  tokenizer at {path.name} has V="
                          f"{self._tok.get_vocab_size()}, need {self.vocab_size} — retraining")
                self._tok = None
        if self._tok is None:
            tok = Tokenizer(models.BPE(unk_token=None))
            tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            tok.decoder = decoders.ByteLevel()
            trainer = trainers.BpeTrainer(
                vocab_size=self.vocab_size, show_progress=verbose,
                initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                special_tokens=["<|endoftext|>"])
            tok.train_from_iterator(text_iter, trainer=trainer)
            path.parent.mkdir(parents=True, exist_ok=True)
            tok.save(str(path))
            self._tok = tok
        self.vocab_size = self._tok.get_vocab_size()
        inv = {i: s for s, i in self._tok.get_vocab().items()}
        # Token strings, for the n-gram codebook: a ByteLevel vocabulary stores
        # the space as 'G-with-dot', so map it back before anything hashes it.
        self.tokens = [inv.get(i, '').replace('\u0120', ' ') for i in range(self.vocab_size)]
        return self

    def encode(self, text):
        return self._tok.encode(text).ids

    def encode_batch(self, texts):
        return [e.ids for e in self._tok.encode_batch(texts)]

    def decode(self, ids):
        return self._tok.decode([int(i) for i in ids])


def build_cache(texts, tokenizer, out_dir, name, val_frac=0.01, verbose=True):
    """Encode `texts` to a `uint16` memmap, returning paths and token counts.

    `texts` is an iterable of strings consumed once. The validation split is the
    tail, so it is the same held-out text for any run against this cache.
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if tokenizer.vocab_size > 65536:
        raise ValueError(f"uint16 cache supports V <= 65536, got {tokenizer.vocab_size}")

    stamp = _stamp(name=name, tok=tokenizer.name, vocab=tokenizer.vocab_size,
                   val_frac=val_frac)
    base = out_dir / f"{name}_{tokenizer.name}{tokenizer.vocab_size}_{stamp[:8]}"
    meta_path = base.with_suffix('.json')
    bin_path = base.with_suffix('.bin')

    # Both files must exist: a tracked or copied `.json` without its `.bin`
    # sidecar is not a cache hit, and `meta['path']` is resolved from the local
    # path rather than trusting whichever OS separator was written into JSON.
    if meta_path.exists() and bin_path.exists() and bin_path.stat().st_size > 0:
        meta = json.loads(meta_path.read_text())
        meta['path'] = bin_path.as_posix()
        if verbose:
            print(f"  cache hit: {meta['train_tokens']:,} train / "
                  f"{meta['val_tokens']:,} val tokens  ({base.name})")
        return meta

    # Partial write under `.part`: an interrupted encode must not leave a
    # truncated cache that the next run reads as complete.
    part = base.with_suffix('.part')
    total, chars = 0, 0
    batch_encode = getattr(tokenizer, 'encode_batch', None)
    try:
        with open(part, 'wb') as fh:
            buf, batch = [], []
            for i, text in enumerate(texts, 1):
                chars += len(text)
                if batch_encode is not None:
                    batch.append(text)
                    if len(batch) >= 2048:
                        for ids in batch_encode(batch):
                            buf.extend(ids)
                        batch.clear()
                else:
                    buf.extend(tokenizer.encode(text))
                if len(buf) >= 1 << 20:
                    np.asarray(buf, dtype=np.uint16).tofile(fh)
                    total += len(buf)
                    buf = []
                if verbose and i % 50000 == 0:
                    print(f"    {i:,} docs, {total + len(buf):,} tokens", flush=True)
            if batch:
                for ids in batch_encode(batch):
                    buf.extend(ids)
            if buf:
                np.asarray(buf, dtype=np.uint16).tofile(fh)
                total += len(buf)
        part.replace(bin_path)
    except BaseException:
        part.unlink(missing_ok=True)
        raise

    n_val = max(1, int(total * val_frac))
    meta = {
        'path': bin_path.as_posix(), 'tokens': total,
        'train_tokens': total - n_val, 'val_tokens': n_val,
        'vocab_size': tokenizer.vocab_size, 'tokenizer': tokenizer.name,
        'chars': chars, 'bytes_per_token': chars / max(total, 1), 'name': name,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    if verbose:
        print(f"  encoded {total:,} tokens from {chars:,} chars "
              f"({meta['bytes_per_token']:.2f} chars/token)")
    return meta


class TokenStream:
    """`B` independent cursors over a token memmap.

    Each row walks its own stretch of the corpus, so the state carried from one
    call to the next belongs to one continuous text -- which is what makes
    truncated BPTT across calls meaningful. `state()` and `load()` carry the
    cursor vector, so a resumed run continues rather than re-reading.
    """

    def __init__(self, meta, batch, seq, split='train', seed=0):
        self.meta, self.batch, self.seq, self.split = meta, batch, seq, split
        self.data = np.memmap(meta['path'], dtype=np.uint16, mode='r')
        n_train = meta['train_tokens']
        self.lo, self.hi = (0, n_train) if split == 'train' else (n_train, meta['tokens'])
        span = self.hi - self.lo
        if span < seq + 1:
            raise ValueError(f"{split} split has {span} tokens, need > seq={seq}")
        rng = np.random.default_rng(seed)
        # Evenly spread, then jittered: no two rows start in the same
        # neighbourhood, and the offsets differ per seed.
        starts = self.lo + (np.arange(batch) * span // batch)
        self.cursor = (starts + rng.integers(0, max(1, span // (batch * 4)), batch)) % (self.hi - seq - 1)
        self.cursor = np.maximum(self.cursor, self.lo)
        self.epoch_tokens = 0

    def next(self):
        """One batch as `(x, y)` int64 arrays of shape `(B, T)`."""
        rows = np.empty((self.batch, self.seq + 1), dtype=np.int64)
        for b, c in enumerate(self.cursor):
            if c + self.seq + 1 > self.hi:
                c = self.lo
            rows[b] = self.data[c:c + self.seq + 1]
            nxt = c + self.seq
            self.cursor[b] = self.lo if nxt + self.seq + 1 > self.hi else nxt
        self.epoch_tokens += self.batch * self.seq
        return rows[:, :-1], rows[:, 1:]

    def state(self):
        return {'cursor': self.cursor.tolist(), 'epoch_tokens': int(self.epoch_tokens)}

    def load(self, state):
        """Restore a saved cursor vector.

        One cursor per batch row, so a resume at a different `--batch` than the
        run was saved at has nothing coherent to restore -- it would read past
        the end of the vector mid-step. Said here rather than as an `IndexError`
        from the next `next()`.
        """
        cursor = np.asarray(state['cursor'], dtype=np.int64)
        if cursor.shape != self.cursor.shape:
            raise ValueError(
                f"saved stream has {cursor.shape[0]} cursors but this stream has "
                f"{self.cursor.shape[0]}: resume with the batch size the "
                f"checkpoint was trained at, or drop --resume")
        self.cursor = cursor
        self.epoch_tokens = state['epoch_tokens']

    @property
    def epochs(self):
        """Passes over the split so far -- the number that says whether a run is
        data-starved rather than over-parameterized."""
        return self.epoch_tokens / max(self.hi - self.lo, 1)


def unigram_counts(meta, vocab_size, limit=20_000_000):
    """Token frequencies over the train split, for the codebook's logit bias."""
    data = np.memmap(meta['path'], dtype=np.uint16, mode='r')
    n = min(meta['train_tokens'], limit)
    return np.bincount(np.asarray(data[:n], dtype=np.int64), minlength=vocab_size)
