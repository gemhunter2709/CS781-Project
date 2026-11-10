"""Arch 1's word-level tokenizer (exactly as in Phase 1) plus an offset-aware encoder for NER."""
import re
import json
from collections import Counter


class CustomTokenizer:
    def __init__(self, vocab_size=32000):
        self.vocab_size = vocab_size
        self.special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]
        self.word2id = {w: i for i, w in enumerate(self.special_tokens)}
        self.id2word = {i: w for i, w in enumerate(self.special_tokens)}
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3
        self.counter = Counter()

    def _tokenize_text(self, text):
        return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

    def encode(self, text, add_special_tokens=False):
        ids = [self.word2id.get(tok, self.unk_id) for tok in self._tokenize_text(text.lower())]
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids, skip_special_tokens=True):
        words = []
        for i in ids:
            if skip_special_tokens and i < len(self.special_tokens):
                continue
            words.append(self.id2word.get(i, "<unk>"))
        out = " ".join(words)
        out = re.sub(r"\s+([.,!?;\)])", r"\1", out)
        out = re.sub(r"(\()\s+", r"\1", out)
        return out

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.word2id = data["word2id"]
        self.id2word = {int(k): v for k, v in data["id2word"].items()}
        self.vocab_size = len(self.word2id)


# Phase 1 used \w+ which glues letters and digits ("Pradesh1", "ENT08"). Splitting them lifts the share of
# gold spans that start and end on a token boundary; both halves are ordinary vocabulary words.
NER_WORD_RE = re.compile(r"[^\W\d_]+|\d+|_+|[^\w\s]", re.UNICODE)


class NerTokenizer(CustomTokenizer):
    split_digits = True

    def encode_with_offsets(self, text):
        splitter = NER_WORD_RE if self.split_digits else re.compile(r"\w+|[^\w\s]", re.UNICODE)
        ids, starts, ends = [], [], []
        for m in splitter.finditer(text):
            ids.append(self.word2id.get(m.group().lower(), self.unk_id))
            starts.append(m.start())
            ends.append(m.end())
        return ids, starts, ends
