"""
Train a RustBPETokenizer (tokenizer.py) on text, then write token_bytes.pt for train_gpt val_bpb.

Requires: rustbpe, tiktoken (see requirements.txt).

Example (Fineweb-style JSONL with a \"text\" field per line):

  python tok_train.py --docs_jsonl ./data/docs_selected.jsonl --tokenizer_dir ./data/tokenizers/my_bpe --vocab_size 8192

Then export .bin shards and train:

  python export_fineweb_custom_bins.py --docs_jsonl ./data/docs_selected.jsonl --tokenizer_dir ./data/tokenizers/my_bpe --output_dir ./data/datasets/my_bpe_dataset

  set DATA_PATH=./data/datasets/my_bpe_dataset
  set TOKENIZER_PATH=./data/tokenizers/my_bpe
  set VOCAB_SIZE=8192
  python train_gpt.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tokenizer import RustBPETokenizer, write_token_mapping_file


def iter_docs_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)["text"]


def main() -> None:
    p = argparse.ArgumentParser(description="Train RustBPETokenizer and write token_bytes.pt")
    p.add_argument("--docs_jsonl", type=Path, required=True, help="JSONL with {\"text\": ...} per line")
    p.add_argument("--tokenizer_dir", type=Path, default=Path("./data/tokenizers/custom_rust_bpe"))
    p.add_argument("--max_chars", type=int, default=10_000_000_000)
    p.add_argument("--vocab_size", type=int, default=8192, help="Full vocab including SPECIAL_TOKENS")
    p.add_argument("--no_chunking", action="store_true", help="BPE on raw bytes (weak regex chunk)")
    p.add_argument("--allow_superchunk", action="store_true")
    p.add_argument("--max_superchunk_chunks", type=int, default=0)
    p.add_argument("--superchunk_pattern", type=str, default=None)
    args = p.parse_args()

    docs_path = args.docs_jsonl.expanduser().resolve()
    if not docs_path.is_file():
        raise FileNotFoundError(docs_path)

    tokenizer_dir = args.tokenizer_dir.expanduser().resolve()
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    def text_iter():
        total_chars = 0
        for text in iter_docs_jsonl(docs_path):
            if total_chars >= args.max_chars:
                break
            rest = args.max_chars - total_chars
            if len(text) > rest:
                text = text[:rest]
            total_chars += len(text)
            yield text

    train_kwargs: dict = {
        "allow_superchunk": args.allow_superchunk,
        "max_superchunk_chunks": args.max_superchunk_chunks,
        "tokenizer_dir": str(tokenizer_dir),
    }
    if args.superchunk_pattern is not None:
        train_kwargs["superchunk_pattern"] = args.superchunk_pattern
    if args.no_chunking:
        logger.info("Training BPE without GPT-4-style regex chunking")
        train_kwargs["chunk_pattern"] = r"[^\n]+"

    t0 = time.time()
    tokenizer = RustBPETokenizer.train_from_iterator(
        text_iter(),
        args.vocab_size,
        **train_kwargs,
    )
    logger.info("Training time: %.2fs", time.time() - t0)

    tokenizer.save(str(tokenizer_dir))
    mapping_path = write_token_mapping_file(str(tokenizer_dir))
    logger.info("Wrote token mapping: %s", mapping_path)

    test_text = """Hello world! This is a test.
Numbers: 123, 4567, 89
Unicode: 你好世界 🌍"""
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    assert decoded == test_text

    vocab_size = tokenizer.get_vocab_size()
    special_set = tokenizer.get_special_tokens()
    token_bytes_list: list[int] = []
    for token_id in range(vocab_size):
        token_str = tokenizer.decode([token_id])
        if token_str in special_set:
            token_bytes_list.append(0)
        else:
            token_bytes_list.append(len(token_str.encode("utf-8")))
    token_bytes = torch.tensor(token_bytes_list, dtype=torch.int32)
    token_bytes_path = tokenizer_dir / "token_bytes.pt"
    torch.save(token_bytes, token_bytes_path)
    logger.info("Saved token_bytes.pt (%d ids) to %s", vocab_size, token_bytes_path)


if __name__ == "__main__":
    main()
