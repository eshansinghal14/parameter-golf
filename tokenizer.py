"""
GPT-4-style BPE tokenizers for nanochat.

Three tokenizer implementations are provided, all sharing a common interface:
- CharLevelTokenizer: byte-level baseline (vocab = 256 bytes + special tokens).
- HuggingFaceTokenizer: BPE training and inference via the HuggingFace tokenizers library.
- RustBPETokenizer: BPE training via rustbpe, inference via tiktoken. (default)

Common interface:
    train_from_iterator(text_iter, vocab_size) -> tokenizer   (classmethod)
    from_directory(path) -> tokenizer                         (classmethod)
    encode(text, prepend=, append=) -> list[int]
    decode(ids) -> str
    get_vocab_size() -> int
    get_special_tokens() -> set[str]
    get_bos_token_id() -> int
    encode_special(name) -> int
    id_to_token(token_id) -> str
    save(directory)

Additionally, RustBPETokenizer provides chat-specific helpers:
    render_conversation(conversation) -> (ids, mask)
    render_for_completion(conversation) -> ids
    visualize_tokenization(ids, mask) -> str
"""

from __future__ import annotations

import os
import copy
from functools import lru_cache
from collections.abc import Iterator
from pathlib import Path

SPECIAL_TOKENS = [
    # every document begins with the Beginning of Sequence (BOS) token that delimits documents
    "<|bos|>",
    # tokens below are only used during finetuning to render Conversations into token ids
    "<|user_start|>", # user messages
    "<|user_end|>",
    "<|assistant_start|>", # assistant messages
    "<|assistant_end|>",
    "<|python_start|>", # assistant invokes python REPL tool
    "<|python_end|>",
    "<|output_start|>", # python REPL outputs back to assistant
    "<|output_end|>",
]

# NOTE: Uses \p{N}{1,2} instead of GPT-4's \p{N}{1,3} to avoid wasting tokens on multi-digit
# numbers at smaller vocab sizes. Empirically validated: BPE with this pattern achieves 0.9433 BPB.
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# -----------------------------------------------------------------------------
# Generic GPT-4-style tokenizer based on HuggingFace Tokenizer
from tokenizers import Tokenizer as HFTokenizer
from tokenizers import pre_tokenizers, decoders, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


class CharLevelTokenizer:
    """
    Character-level (byte-level) tokenizer.
    
    Maps each byte (0-255) to a token ID, plus special tokens.
    Vocab size = 256 (bytes) + len(SPECIAL_TOKENS)
    
    This is the simplest possible tokenizer - useful as a baseline for
    comparing against BPE tokenizers with various vocabulary sizes.
    """
    
    def __init__(self) -> None:
        # Build special tokens mapping: special tokens come after the 256 byte tokens
        self._special_tokens: dict[str, int] = {name: 256 + i for i, name in enumerate(SPECIAL_TOKENS)}
        self._special_tokens_reverse: dict[int, str] = {v: k for k, v in self._special_tokens.items()}
        self._vocab_size: int = 256 + len(SPECIAL_TOKENS)
        self.bos_token_id: int = self._special_tokens["<|bos|>"]

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size: int | None = None) -> "CharLevelTokenizer":
        """
        'Training' a char-level tokenizer is a no-op since it's just bytes.
        vocab_size argument is ignored (always 256 + special tokens).
        """
        if vocab_size is not None and vocab_size != 256 + len(SPECIAL_TOKENS):
            print(f"Warning: CharLevelTokenizer ignores vocab_size={vocab_size}, using {256 + len(SPECIAL_TOKENS)}")
        return cls()
    
    @classmethod
    def from_directory(cls, tokenizer_dir: str) -> "CharLevelTokenizer":
        """Load from directory - just checks the marker file exists."""
        marker_path = os.path.join(tokenizer_dir, "char_tokenizer.marker")
        if not os.path.exists(marker_path):
            raise FileNotFoundError(f"CharLevelTokenizer marker not found at {marker_path}")
        return cls()

    def get_vocab_size(self) -> int:
        return self._vocab_size

    def get_special_tokens(self) -> set[str]:
        return set(self._special_tokens.keys())

    def id_to_token(self, token_id: int) -> str:
        if token_id < 256:
            # Byte token - return the character representation
            return bytes([token_id]).decode('utf-8', errors='replace')
        elif token_id in self._special_tokens_reverse:
            return self._special_tokens_reverse[token_id]
        else:
            raise ValueError(f"Invalid token id: {token_id}")
    
    @lru_cache(maxsize=32)
    def encode_special(self, text: str) -> int:
        """Encode a special token by exact match."""
        if text not in self._special_tokens:
            raise ValueError(f"Unknown special token: {text}")
        return self._special_tokens[text]

    def get_bos_token_id(self) -> int:
        return self.bos_token_id

    def _encode_bytes(self, text: str) -> list[int]:
        """Encode text to byte token IDs."""
        return list(text.encode('utf-8'))

    def encode(self, text: str | list[str], prepend: str | int | None = None, append: str | int | None = None, num_threads: int | None = None) -> list[int] | list[list[int]]:
        """
        Encode text to token IDs.
        text can be a string or a list of strings.
        """
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)
        
        if isinstance(text, str):
            ids = self._encode_bytes(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
            if append is not None:
                ids.append(append_id)
        elif isinstance(text, list):
            ids = [self._encode_bytes(t) for t in text]
            if prepend is not None:
                for ids_row in ids:
                    ids_row.insert(0, prepend_id)
            if append is not None:
                for ids_row in ids:
                    ids_row.append(append_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
        
        return ids
    
    def __call__(self, *args, **kwargs) -> list[int] | list[list[int]]:
        return self.encode(*args, **kwargs)

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to text."""
        byte_ids = []
        result_parts = []
        
        for token_id in ids:
            if token_id < 256:
                byte_ids.append(token_id)
            else:
                # Flush accumulated bytes
                if byte_ids:
                    result_parts.append(bytes(byte_ids).decode('utf-8', errors='replace'))
                    byte_ids = []
                # Add special token
                if token_id in self._special_tokens_reverse:
                    result_parts.append(self._special_tokens_reverse[token_id])
                else:
                    result_parts.append(f"<|unk_{token_id}|>")
        
        # Flush remaining bytes
        if byte_ids:
            result_parts.append(bytes(byte_ids).decode('utf-8', errors='replace'))
        
        return ''.join(result_parts)
    
    def save(self, tokenizer_dir: str) -> None:
        """Save the tokenizer to disk."""
        import torch
        os.makedirs(tokenizer_dir, exist_ok=True)
        
        # Write a marker file so we know this is a char-level tokenizer
        marker_path = os.path.join(tokenizer_dir, "char_tokenizer.marker")
        with open(marker_path, "w") as f:
            f.write("CharLevelTokenizer\n")
            f.write(f"vocab_size={self._vocab_size}\n")
        
        # Also save token_bytes.pt for compatibility with the training pipeline
        # Each token is just its byte value (0-255), special tokens get placeholder bytes
        token_bytes_list = []
        max_len = max(len(name.encode('utf-8')) for name in SPECIAL_TOKENS)
        max_len = max(max_len, 1)  # At least 1 byte per token
        
        for i in range(self._vocab_size):
            if i < 256:
                # Byte token
                token_bytes_list.append([i] + [0] * (max_len - 1))
            else:
                # Special token - encode its name
                special_name = self._special_tokens_reverse[i]
                encoded = list(special_name.encode('utf-8'))
                # Pad to max_len
                encoded = encoded[:max_len] + [0] * (max_len - len(encoded))
                token_bytes_list.append(encoded)
        
        token_bytes = torch.tensor(token_bytes_list, dtype=torch.uint8)
        token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
        torch.save(token_bytes, token_bytes_path)
        
        print(f"Saved CharLevelTokenizer to {tokenizer_dir}")
        print(f"  - marker: {marker_path}")
        print(f"  - token_bytes: {token_bytes_path} (shape: {token_bytes.shape})")


class HuggingFaceTokenizer:
    """
    BPE tokenizer backed by the HuggingFace tokenizers library.

    Supports both training from raw text and loading a previously trained
    tokenizer from disk. Uses GPT-4-style byte-level BPE with the same
    split pattern and special tokens as the other nanochat tokenizers.
    """

    def __init__(self, tokenizer: HFTokenizer) -> None:
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, hf_path: str) -> "HuggingFaceTokenizer":
        tokenizer = HFTokenizer.from_pretrained(hf_path)
        return cls(tokenizer)

    @classmethod
    def from_directory(cls, tokenizer_dir: str) -> "HuggingFaceTokenizer":
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        tokenizer = HFTokenizer.from_file(tokenizer_path)
        return cls(tokenizer)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size: int) -> "HuggingFaceTokenizer":
        tokenizer = HFTokenizer(BPE(
            byte_fallback=True, # needed!
            unk_token=None,
            fuse_unk=False,
        ))
        tokenizer.normalizer = None
        # NOTE: Uses \p{N}{1,2} instead of GPT-4's \p{N}{1,3} to avoid wasting tokens on multi-digit
        # numbers at smaller vocab sizes. Empirically validated: BPE with this pattern achieves 0.9433 BPB.
        gpt4_split_regex = Regex(SPLIT_PATTERN) # huggingface demands that you wrap it in Regex!!
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(pattern=gpt4_split_regex, behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
        ])
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = None
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            show_progress=True,
            min_frequency=0,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=SPECIAL_TOKENS,
        )
        tokenizer.train_from_iterator(text_iterator, trainer)
        return cls(tokenizer)

    def get_vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def get_special_tokens(self) -> set[str]:
        special_tokens_map = self.tokenizer.get_added_tokens_decoder()
        return {w.content for w in special_tokens_map.values()}

    def id_to_token(self, token_id: int) -> str:
        return self.tokenizer.id_to_token(token_id)

    def _encode_one(self, text: str, prepend: str | int | None = None, append: str | int | None = None) -> list[int]:
        # prepend/append can be either a string of a special token or a token id directly.
        assert isinstance(text, str)
        ids = []
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
            ids.append(prepend_id)
        ids.extend(self.tokenizer.encode(text, add_special_tokens=False).ids)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)
            ids.append(append_id)
        return ids

    def encode_special(self, text: str) -> int:
        return self.tokenizer.token_to_id(text)

    def get_bos_token_id(self) -> int:
        return self.encode_special("<|bos|>")

    def encode(self, text: str | list[str], *args, **kwargs) -> list[int] | list[list[int]]:
        if isinstance(text, str):
            return self._encode_one(text, *args, **kwargs)
        elif isinstance(text, list):
            return [self._encode_one(t, *args, **kwargs) for t in text]
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

    def __call__(self, *args, **kwargs) -> list[int] | list[list[int]]:
        return self.encode(*args, **kwargs)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def save(self, tokenizer_dir: str) -> None:
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        self.tokenizer.save(tokenizer_path)
        print(f"Saved tokenizer to {tokenizer_path}")

# -----------------------------------------------------------------------------
# Tokenizer based on rustbpe + tiktoken combo
import pickle
import rustbpe
import tiktoken

class RustBPETokenizer:
    """
    BPE tokenizer: trains with rustbpe, runs inference with tiktoken.

    This is the default tokenizer for nanochat. Training is handled by rustbpe
    (a fast Rust BPE implementation), and the resulting merge table is wrapped
    in a tiktoken Encoding for efficient inference. Also provides chat-specific
    methods for rendering conversations into token sequences with supervision masks.
    """

    def __init__(self, enc: tiktoken.Encoding, bos_token: str) -> None:
        self.enc = enc
        self.bos_token_id: int = self.encode_special(bos_token)

    @classmethod
    def train_from_iterator(
        cls,
        text_iterator: Iterator[str],
        vocab_size: int,
        allow_superchunk: bool,
        pattern: str = SPLIT_PATTERN,
        max_superchunk_chunks: int = 0,
        tokenizer_dir: str | None = None,
        chunk_pattern: str | None = None,
        superchunk_pattern: str | None = None,
    ) -> "RustBPETokenizer":
        # 1) train using rustbpe
        tokenizer = rustbpe.Tokenizer()
        # the special tokens are inserted later in __init__, we don't train them here
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        assert vocab_size_no_special >= 256, f"vocab_size_no_special must be at least 256, got {vocab_size_no_special}"
        # rustbpe uses chunk_pattern for the split regex (no longer "pattern")
        effective_chunk_pattern = chunk_pattern if chunk_pattern is not None else pattern
        train_kwargs = {
            "vocab_size": vocab_size_no_special,
            "chunk_pattern": effective_chunk_pattern,
            "allow_superchunk": allow_superchunk,
            "max_superchunk_chunks": max_superchunk_chunks,
        }
        if tokenizer_dir is not None:
            train_kwargs["tokenizer_dir"] = tokenizer_dir
        if superchunk_pattern is not None:
            train_kwargs["superchunk_pattern"] = superchunk_pattern
        tokenizer.train_from_iterator(text_iterator, **train_kwargs)
        # 2) construct the associated tiktoken encoding for inference
        pattern = tokenizer.get_pattern()
        mergeable_ranks_list = tokenizer.get_mergeable_ranks()
        mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
        tokens_offset = len(mergeable_ranks)
        special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
        enc = tiktoken.Encoding(
            name="rustbpe",
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks, # dict[bytes, int] (token bytes -> merge priority rank)
            special_tokens=special_tokens, # dict[str, int] (special token name -> token id)
        )
        return cls(enc, "<|bos|>")

    @classmethod
    def from_directory(cls, tokenizer_dir: str) -> "RustBPETokenizer":
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "rb") as f:
            enc = pickle.load(f)
        return cls(enc, "<|bos|>")

    @classmethod
    def from_pretrained(cls, tiktoken_name: str) -> "RustBPETokenizer":
        # https://github.com/openai/tiktoken/blob/eedc8563/tiktoken_ext/openai_public.py
        enc = tiktoken.get_encoding(tiktoken_name)
        # tiktoken calls the special document delimiter token "<|endoftext|>"
        # yes this is confusing because this token is almost always PREPENDED to the beginning of the document
        # it most often is used to signal the start of a new sequence to the LLM during inference etc.
        # so in nanoChat we always use "<|bos|>" short for "beginning of sequence", but historically it is often called "<|endoftext|>".
        return cls(enc, "<|endoftext|>")

    def get_vocab_size(self) -> int:
        return self.enc.n_vocab

    def get_special_tokens(self) -> set[str]:
        return self.enc.special_tokens_set

    def id_to_token(self, token_id: int) -> str:
        return self.enc.decode([token_id])

    @lru_cache(maxsize=32)
    def encode_special(self, text: str) -> int:
        return self.enc.encode_single_token(text)

    def get_bos_token_id(self) -> int:
        return self.bos_token_id

    def encode(self, text: str | list[str], prepend: str | int | None = None, append: str | int | None = None, num_threads: int = 8) -> list[int] | list[list[int]]:
        """
        Encode text to token IDs. Accepts a string or list of strings.
        prepend/append: optional special token name (str) or token id (int)
        to insert at the start/end of each encoded sequence.
        """

        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)

        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
            if append is not None:
                ids.append(append_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for ids_row in ids:
                    ids_row.insert(0, prepend_id)
            if append is not None:
                for ids_row in ids:
                    ids_row.append(append_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

        return ids

    def __call__(self, *args, **kwargs) -> list[int] | list[list[int]]:
        return self.encode(*args, **kwargs)

    def decode(self, ids: list[int]) -> str:
        return self.enc.decode(ids)

    def set_chunking(self, enabled: bool) -> None:
        """Enable/disable GPT-4-style regex chunking.

        When disabled, replaces the split pattern with a catch-all so tiktoken
        processes raw bytes without pre-splitting into words/numbers/etc.
        """
        if not enabled:
            # Rebuild encoding with catch-all pattern (no chunking)
            self.enc = tiktoken.Encoding(
                name=self.enc.name + "_unchunked",
                pat_str=r"[\s\S]+",
                mergeable_ranks=self.enc._mergeable_ranks,
                special_tokens=self.enc._special_tokens,
            )
        # enabled=True is the default (BPE already uses chunking), nothing to do

    def save(self, tokenizer_dir: str) -> None:
        os.makedirs(tokenizer_dir, exist_ok=True)
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(self.enc, f)
        print(f"Saved tokenizer encoding to {pickle_path}")

    def render_conversation(self, conversation: dict, max_tokens: int = 2048) -> tuple[list[int], list[int]]:
        """
        Tokenize a single Chat conversation (which we call a "doc" or "document" here).
        Returns:
        - ids: list[int] is a list of token ids of this rendered conversation
        - mask: list[int] of same length, mask = 1 for tokens that the Assistant is expected to train on.
        """
        # ids, masks that we will return and a helper function to help build them up.
        ids, mask = [], []
        def add_tokens(token_ids: int | list[int], mask_val: int) -> None:
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))

        # sometimes the first message is a system message...
        # => just merge it with the second (user) message
        if conversation["messages"][0]["role"] == "system":
            # some conversation surgery is necessary here for now...
            conversation = copy.deepcopy(conversation) # avoid mutating the original
            messages = conversation["messages"]
            assert messages[1]["role"] == "user", "System message must be followed by a user message"
            messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
            messages = messages[1:]
        else:
            messages = conversation["messages"]
        assert len(messages) >= 1, f"Conversation has less than 1 message: {messages}"

        # fetch all the special tokens we need
        bos = self.get_bos_token_id()
        user_start, user_end = self.encode_special("<|user_start|>"), self.encode_special("<|user_end|>")
        assistant_start, assistant_end = self.encode_special("<|assistant_start|>"), self.encode_special("<|assistant_end|>")
        python_start, python_end = self.encode_special("<|python_start|>"), self.encode_special("<|python_end|>")
        output_start, output_end = self.encode_special("<|output_start|>"), self.encode_special("<|output_end|>")

        # now we can tokenize the conversation
        add_tokens(bos, 0)
        for i, message in enumerate(messages):

            # some sanity checking here around assumptions, to prevent footguns
            must_be_from = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == must_be_from, f"Message {i} is from {message['role']} but should be from {must_be_from}"

            # content can be either a simple string or a list of parts (e.g. containing tool calls)
            content = message["content"]

            if message["role"] == "user":
                assert isinstance(content, str), "User messages are simply expected to be strings"
                value_ids = self.encode(content)
                add_tokens(user_start, 0)
                add_tokens(value_ids, 0)
                add_tokens(user_end, 0)
            elif message["role"] == "assistant":
                add_tokens(assistant_start, 0)
                if isinstance(content, str):
                    # simple string => simply add the tokens
                    value_ids = self.encode(content)
                    add_tokens(value_ids, 1)
                elif isinstance(content, list):
                    for part in content:
                        value_ids = self.encode(part["text"])
                        if part["type"] == "text":
                            # string part => simply add the tokens
                            add_tokens(value_ids, 1)
                        elif part["type"] == "python":
                            # python tool call => add the tokens inside <|python_start|> and <|python_end|>
                            add_tokens(python_start, 1)
                            add_tokens(value_ids, 1)
                            add_tokens(python_end, 1)
                        elif part["type"] == "python_output":
                            # python output => add the tokens inside <|output_start|> and <|output_end|>
                            # none of these tokens are supervised because the tokens come from Python at test time
                            add_tokens(output_start, 0)
                            add_tokens(value_ids, 0)
                            add_tokens(output_end, 0)
                        else:
                            raise ValueError(f"Unknown part type: {part['type']}")
                else:
                    raise ValueError(f"Unknown content type: {type(content)}")
                add_tokens(assistant_end, 1)

        # truncate to max_tokens tokens MAX (helps prevent OOMs)
        ids = ids[:max_tokens]
        mask = mask[:max_tokens]
        return ids, mask

    def visualize_tokenization(self, ids: list[int], mask: list[int], with_token_id: bool = False) -> str:
        """
        Debug helper: colorize tokens from render_conversation.
        Green = supervised (mask=1), Red = unsupervised (mask=0).
        """
        RED = '\033[91m'
        GREEN = '\033[92m'
        RESET = '\033[0m'
        GRAY = '\033[90m'
        tokens = []
        for i, (token_id, mask_val) in enumerate(zip(ids, mask)):
            token_str = self.decode([token_id])
            color = GREEN if mask_val == 1 else RED
            tokens.append(f"{color}{token_str}{RESET}")
            if with_token_id:
                tokens.append(f"{GRAY}({token_id}){RESET}")
        return '|'.join(tokens)

    def render_for_completion(self, conversation: dict) -> list[int]:
        """
        Used during Reinforcement Learning. In that setting, we want to
        render the conversation priming the Assistant for a completion.
        Unlike the Chat SFT case, we don't need to return the mask.
        """
        # We have some surgery to do: we need to pop the last message (of the Assistant)
        conversation = copy.deepcopy(conversation) # avoid mutating the original
        messages = conversation["messages"]
        assert messages[-1]["role"] == "assistant", "Last message must be from the Assistant"
        messages.pop() # remove the last message (of the Assistant) inplace

        # Now tokenize the conversation
        ids, mask = self.render_conversation(conversation)

        # Finally, to prime the Assistant for a completion, append the Assistant start token
        assistant_start = self.encode_special("<|assistant_start|>")
        ids.append(assistant_start)
        return ids

# -----------------------------------------------------------------------------
# nanochat-specific convenience functions

def _default_tokenizer_dir() -> str:
    return str(Path(__file__).resolve().parent / "tokenizer")


def get_tokenizer(tokenizer_path: str | None = None) -> RustBPETokenizer:
    """Load RustBPETokenizer from disk. Default: ./tokenizer next to this module."""
    tokenizer_dir = tokenizer_path if tokenizer_path is not None else _default_tokenizer_dir()
    return RustBPETokenizer.from_directory(tokenizer_dir)

def token_id_to_token_str_map(tokenizer_dir: str) -> dict[int, str]:
    """Load a tokenizer folder and return a mapping from token id -> token string.

    Supported formats:
    - RustBPETokenizer: <tokenizer_dir>/tokenizer.pkl
    - HuggingFaceTokenizer: <tokenizer_dir>/tokenizer.json
    """
    rust_pkl_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
    hf_json_path = os.path.join(tokenizer_dir, "tokenizer.json")

    if os.path.exists(rust_pkl_path):
        tok = RustBPETokenizer.from_directory(tokenizer_dir)
        id_to_str: dict[int, str] = {}
        n_vocab = tok.enc.n_vocab
        for token_id in range(n_vocab):
            try:
                token_bytes = tok.enc.decode_single_token_bytes(token_id)
            except Exception:
                # Fall back to slower but more general decoding.
                try:
                    id_to_str[token_id] = tok.decode([token_id])
                    continue
                except Exception:
                    id_to_str[token_id] = f"<decode_error:{token_id}>"
                    continue

            try:
                id_to_str[token_id] = token_bytes.decode("utf-8")
            except UnicodeDecodeError:
                # Keep it readable and stable for arbitrary byte-level tokens.
                id_to_str[token_id] = token_bytes.decode("utf-8", errors="backslashreplace")
        return id_to_str

    if os.path.exists(hf_json_path):
        tok = HuggingFaceTokenizer.from_directory(tokenizer_dir)
        vocab = tok.tokenizer.get_vocab()  # token_str -> token_id
        id_to_str = {token_id: token_str for token_str, token_id in vocab.items()}
        return id_to_str

    raise FileNotFoundError(
        f"No supported tokenizer files found in {tokenizer_dir}. "
        "Expected tokenizer.pkl (RustBPETokenizer) or tokenizer.json (HuggingFaceTokenizer)."
    )

def write_token_mapping_file(tokenizer_dir: str, filename: str = "token_mapping") -> str:
    id_to_str = token_id_to_token_str_map(tokenizer_dir)
    out_path = os.path.join(tokenizer_dir, filename)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for token_id in range(len(id_to_str)):
            token_str = id_to_str[token_id]
            token_str = token_str.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n")
            f.write(f"{token_id}\t{token_str}\n")
    return out_path

def get_token_bytes(device: str = "cpu", tokenizer_path: str | None = None):
    """Load the token_bytes.pt tensor (written by tok_train.py) for BPB / visualization."""
    import torch

    tokenizer_dir = tokenizer_path if tokenizer_path is not None else _default_tokenizer_dir()
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    assert os.path.exists(token_bytes_path), f"Token bytes not found at {token_bytes_path}? It gets written by tok_train.py"
    with open(token_bytes_path, "rb") as f:
        token_bytes = torch.load(f, map_location=device, weights_only=True)
    return token_bytes
