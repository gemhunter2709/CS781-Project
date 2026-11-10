"""
Byte-level BPE tokenizer, written from scratch (stdlib only for the tokenizer logic).

How it works, in short:
  1. A regex chops text into "pre-tokens" (a word with its leading space, a run of digits,
     a run of punctuation, ...). Merges never cross these boundaries.
  2. Every pre-token is turned into raw UTF-8 bytes (ids 0..255), so *any* text can be
     encoded - there is no <unk> and no lowercasing.
  3. Training repeatedly merges the most frequent adjacent id pair into a new id.
     We train on the table of *unique* pre-tokens with their counts, which is what keeps
     it fast in pure Python.
  4. Encoding replays the learned merges (lowest rank first) per pre-token, and caches the
     result, so repeated words cost a single dict lookup.
"""
import re
import json
import heapq
from collections import Counter, defaultdict

# Close to the GPT-2 splitter, but built on plain `re` (no \p{L}), so it needs no extra package.
PRETOKEN_RE = re.compile(
    r"'(?:s|t|re|ve|m|ll|d)| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+",
    re.UNICODE,
)

EOS_TOKEN = "<|endoftext|>"
PAD_TOKEN = "<|pad|>"


class ByteBPE:
    def __init__(self, merges=None, vocab_size=32000):
        self.vocab_size = vocab_size
        self.merges = [tuple(m) for m in (merges or [])]      # merge i creates id 256 + i
        self.eos_id = vocab_size - 2
        self.pad_id = vocab_size - 1
        self._cache = {}
        self._rebuild_tables()

    # ------------------------------------------------------------- internals
    def _rebuild_tables(self):
        self.ranks = {pair: (i, 256 + i) for i, pair in enumerate(self.merges)}
        vocab = [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            vocab.append(vocab[a] + vocab[b])
        self.vocab = vocab
        self._cache.clear()

    # -------------------------------------------------------------- training
    @classmethod
    def train(cls, texts, vocab_size=32000, log_every=2000):
        """Learn merges from an iterable of strings."""
        n_merges = vocab_size - 256 - 2          # leave room for the two special tokens
        word_counts = Counter()
        for text in texts:
            word_counts.update(PRETOKEN_RE.findall(text))
        print(f"  unique pre-tokens: {len(word_counts):,}")

        words = [list(w.encode("utf-8")) for w in word_counts]
        freqs = list(word_counts.values())
        del word_counts

        pair_count = defaultdict(int)
        where = defaultdict(set)                 # pair -> indices of words that (may) contain it
        for wi, w in enumerate(words):
            f = freqs[wi]
            for pair in zip(w, w[1:]):
                pair_count[pair] += f
                where[pair].add(wi)

        # Max-heap with lazy invalidation: stale entries are fixed up when they surface.
        heap = [(-c, p) for p, c in pair_count.items()]
        heapq.heapify(heap)

        merges = []
        vocab = [bytes([i]) for i in range(256)]
        for m in range(n_merges):
            best = None
            while heap:
                neg, pair = heapq.heappop(heap)
                real = pair_count.get(pair, 0)
                if real <= 0:
                    continue
                if real != -neg:
                    heapq.heappush(heap, (-real, pair))
                    continue
                best = pair
                break
            if best is None:
                print(f"  ran out of pairs after {m} merges")
                break

            new_id = 256 + m
            merges.append(best)
            a, b = best
            vocab.append(vocab[a] + vocab[b])
            touched = {}
            for wi in where.pop(best, ()):
                w = words[wi]
                f = freqs[wi]
                # find the occurrences in this word (left to right, non-overlapping)
                if len(w) < 2:
                    continue
                i, out, changed = 0, [], False
                while i < len(w):
                    if i < len(w) - 1 and w[i] == a and w[i + 1] == b:
                        out.append(new_id)
                        i += 2
                        changed = True
                    else:
                        out.append(w[i])
                        i += 1
                if not changed:
                    continue                      # stale entry in `where`
                for pair in zip(w, w[1:]):        # take the old word's pairs away ...
                    pair_count[pair] -= f
                for pair in zip(out, out[1:]):    # ... and add the new word's pairs
                    pair_count[pair] += f
                    where[pair].add(wi)
                    touched[pair] = pair_count[pair]
                words[wi] = out
            for pair, c in touched.items():
                if c > 0:
                    heapq.heappush(heap, (-c, pair))
            pair_count.pop(best, None)

            if (m + 1) % log_every == 0:
                print(f"  merge {m + 1:,}/{n_merges:,}  last = {vocab[-1]!r}")

        return cls(merges, vocab_size)

    # -------------------------------------------------------------- encoding
    def _encode_word(self, word):
        ids = list(word.encode("utf-8"))
        ranks = self.ranks
        while len(ids) > 1:
            best_rank, best_pair, best_new = None, None, None
            for pair in zip(ids, ids[1:]):
                hit = ranks.get(pair)
                if hit is not None and (best_rank is None or hit[0] < best_rank):
                    best_rank, best_pair, best_new = hit[0], pair, hit[1]
            if best_pair is None:
                break
            a, b = best_pair
            out, i = [], 0
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == a and ids[i + 1] == b:
                    out.append(best_new)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            ids = out
        return ids

    def encode(self, text):
        cache = self._cache
        if len(cache) > 1_000_000:               # keep RAM bounded on huge corpora
            cache.clear()
        out = []
        for word in PRETOKEN_RE.findall(text):
            ids = cache.get(word)
            if ids is None:
                ids = cache[word] = self._encode_word(word)
            out.extend(ids)
        return out

    def decode(self, ids):
        specials = (self.eos_id, self.pad_id)
        data = b"".join(self.vocab[i] for i in ids if i < len(self.vocab) and i not in specials)
        return data.decode("utf-8", errors="replace")

    # ----------------------------------------------------------- persistence
    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"vocab_size": self.vocab_size, "merges": self.merges,
                       "eos": EOS_TOKEN, "pad": PAD_TOKEN}, f)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            blob = json.load(f)
        return cls(blob["merges"], blob["vocab_size"])

