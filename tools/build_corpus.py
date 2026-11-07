"""Build the Aleph token cache from TinyStories.

TinyStories (Eldan & Li, arXiv:2305.07759) is the corpus that makes a tiny
model worth measuring: vocabulary and syntax restricted to what a young child
uses, so a few-million-parameter model produces fluent, coherent English where
the same model on web text produces mush. A parameter-efficiency claim needs a
corpus where small models are *supposed* to work, or the score measures the
corpus's difficulty rather than the architecture.

Run once per (tokenizer, vocab) pair; the cache is stamped and reused after.
"""

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from aleph.data.stream import BPETokenizer, ByteTokenizer, build_cache  # noqa: E402


def iter_texts(limit=None, verbose=True):
    """Stream TinyStories rows, so nothing larger than one shard is in memory."""
    from datasets import load_dataset
    ds = load_dataset('roneneldan/TinyStories', split='train', streaming=True)
    t0 = time.perf_counter()
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        if verbose and i and i % 100_000 == 0:
            print(f"    read {i:,} stories in {time.perf_counter() - t0:.0f}s", flush=True)
        # One document per story, terminated: the model has to learn where a
        # story ends, and an unterminated stream would teach it stories never do.
        yield row['text'].strip() + "\n<|endoftext|>\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tokenizer', choices=('bpe', 'byte'), default='bpe')
    p.add_argument('--vocab', type=int, default=4096,
                   help="BPE vocabulary size. Nearly free under the codebook, "
                        "so chosen for sequence length rather than parameters.")
    p.add_argument('--stories', type=int, default=400_000,
                   help="Stories to encode. 400k is ~100M tokens, enough that a "
                        "sub-1M-parameter model is not the thing running out.")
    p.add_argument('--train-tokenizer-on', type=int, default=60_000,
                   help="Stories used to fit the BPE merges.")
    p.add_argument('--out', default='data')
    a = p.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if a.tokenizer == 'byte':
        tok = ByteTokenizer()
        print(f"byte tokenizer: V={tok.vocab_size}, no table to fit")
    else:
        tok = BPETokenizer(a.vocab)
        path = out / f"bpe_{a.vocab}.json"
        print(f"fitting BPE V={a.vocab} on {a.train_tokenizer_on:,} stories...")
        t0 = time.perf_counter()
        tok.train(iter_texts(a.train_tokenizer_on, verbose=False), path)
        print(f"  done in {time.perf_counter() - t0:.0f}s -> {path.name}")

    print(f"encoding {a.stories:,} stories...")
    meta = build_cache(iter_texts(a.stories), tok, out, f"tinystories{a.stories // 1000}k")
    print(f"\ncache: {meta['path']}")
    print(f"  train {meta['train_tokens']:,} / val {meta['val_tokens']:,} tokens")
    print(f"  {meta['bytes_per_token']:.2f} chars per token  (V={meta['vocab_size']})")
    # The figure a bits/byte score is divided by, so it belongs in the log.
    print(f"  a 1M-token budget is {meta['train_tokens'] / 1e6:.1f} epochs of this split")


if __name__ == '__main__':
    main()
