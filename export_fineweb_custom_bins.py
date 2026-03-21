"""
Export fineweb_train_*.bin / fineweb_val_*.bin using tokenizer.RustBPETokenizer (tokenizer.pkl).

Same on-disk layout as data/download_hf_docs_and_tokenize.py (header magic 20240520, uint16 tokens).
First NUM_VAL_DOCS documents go to val shards (default 50_000), same as the official pipeline.

Usage:

  python export_fineweb_custom_bins.py \\
    --docs_jsonl ./data/docs_selected.jsonl \\
    --tokenizer_dir ./data/tokenizers/my_bpe \\
    --output_dir ./data/datasets/my_bpe_dataset
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tokenizer import RustBPETokenizer

# Match data/download_hf_docs_and_tokenize.py
NUM_VAL_DOCS = 50_000
SHARD_SIZE = 10**8
DATAFILE_MAGIC = 20240520
DATAFILE_VERSION = 1


def write_datafile(path: Path, toks: Any) -> None:
    if len(toks) >= 2**31:
        raise ValueError("token count too large")
    header = np.zeros(256, dtype="<i4")
    header[0] = DATAFILE_MAGIC
    header[1] = DATAFILE_VERSION
    header[2] = len(toks)
    toks = np.asarray(toks)
    if toks.dtype != np.uint16:
        if not ((0 <= toks).all() and (toks < 2**16).all()):
            raise ValueError("token dictionary too large for uint16")
        toks = toks.astype("<u2", copy=False)
    else:
        toks = toks.astype("<u2", copy=False)
    with path.open("wb") as f:
        f.write(header.tobytes())
        f.write(toks.tobytes())


def iter_docs(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)["text"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs_jsonl", type=Path, required=True)
    ap.add_argument("--tokenizer_dir", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--num_val_docs", type=int, default=NUM_VAL_DOCS)
    ap.add_argument("--shard_size", type=int, default=SHARD_SIZE)
    args = ap.parse_args()

    tok = RustBPETokenizer.from_directory(str(args.tokenizer_dir.expanduser().resolve()))
    vocab_size = tok.get_vocab_size()
    if vocab_size > 2**16:
        raise ValueError(f"vocab_size={vocab_size} does not fit uint16 shards")
    bos_id = tok.get_bos_token_id()

    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    for pattern in ("fineweb_train_*.bin", "fineweb_val_*.bin"):
        for stale in out.glob(pattern):
            stale.unlink()

    buf = np.empty((args.shard_size,), dtype=np.uint16)
    fill = 0
    split = "val"
    shards = {"val": 0, "train": 0}
    docs_total = 0

    def flush() -> None:
        nonlocal fill
        if fill == 0:
            return
        write_datafile(out / f"fineweb_{split}_{shards[split]:06d}.bin", buf[:fill])
        shards[split] += 1
        fill = 0

    for text in iter_docs(args.docs_jsonl.expanduser().resolve()):
        split_for_doc = "val" if docs_total < args.num_val_docs else "train"
        if split_for_doc != split:
            flush()
            split = split_for_doc

        body = tok.encode(text)
        n = len(body) + 1
        toks = np.empty((n,), dtype=np.int32)
        toks[0] = bos_id
        toks[1:] = body
        if not ((0 <= toks).all() and (toks < vocab_size).all()):
            bad = int(toks[(toks < 0) | (toks >= vocab_size)][0])
            raise ValueError(f"token id {bad} outside vocab_size={vocab_size}")

        toks_u2 = toks.astype("<u2", copy=False)
        pos = 0
        while pos < len(toks_u2):
            take = min(args.shard_size - fill, len(toks_u2) - pos)
            buf[fill : fill + take] = toks_u2[pos : pos + take]
            fill += take
            pos += take
            if fill == args.shard_size:
                flush()
        docs_total += 1
        if docs_total % 100_000 == 0:
            print(f"{out.name}: {docs_total} docs", flush=True)

    flush()
    print(f"Done. Wrote shards under {out} ({docs_total} docs).")


if __name__ == "__main__":
    main()
