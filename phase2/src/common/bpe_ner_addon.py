

# ---------------------------------------------------------------------------------------------------
# Phase 2 addition: encode with character offsets, so predicted tokens can be mapped back to exact spans.
# ---------------------------------------------------------------------------------------------------
# Same splitter as above, except a run of punctuation is cut into single characters. Gold entities often end
# in the middle of runs like "Ors.[1]" or "Cr.P.C.,"; with this split 99.3% of gold spans start and end on a
# token boundary (98.1% without it). Every piece is still encoded with the Phase-1 merges.
NER_PRETOKEN_RE = re.compile(
    r"'(?:s|t|re|ve|m|ll|d)| ?[^\W\d_]+| ?\d+| ?[^\s\w]|\s+(?!\S)|\s+",
    re.UNICODE,
)


class NerTokenizer(ByteBPE):
    split_punct = True

    def encode_with_offsets(self, text):
        """Returns ids and, per token, the [start, end) of its non-whitespace characters in `text`."""
        splitter = NER_PRETOKEN_RE if self.split_punct else PRETOKEN_RE
        ids, starts, ends = [], [], []
        for m in splitter.finditer(text):
            word, base = m.group(), m.start()
            toks = self._cache.get(word)
            if toks is None:
                toks = self._cache[word] = self._encode_word(word)
            # byte position -> character position inside this piece (pieces can contain multi-byte characters)
            char_at_byte = []
            for ci, ch in enumerate(word):
                char_at_byte += [ci] * len(ch.encode("utf-8"))
            b = 0
            for t in toks:
                n = len(self.vocab[t])
                cs, ce = char_at_byte[b], char_at_byte[b + n - 1] + 1
                b += n
                piece = word[cs:ce]
                s = base + cs + (len(piece) - len(piece.lstrip()))
                e = base + ce - (len(piece) - len(piece.rstrip()))
                if s >= e:
                    s = e = base + ce
                ids.append(t)
                starts.append(s)
                ends.append(e)
        return ids, starts, ends
