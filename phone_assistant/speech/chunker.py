"""Splits streamed LLM text into speakable chunks.

The first chunk is cut early (at a clause boundary once a few words exist) so the caller hears
speech as soon as possible; later chunks are whole sentences so the voice sounds continuous.
"""
from __future__ import annotations

import re

_SENT_END = re.compile(r"([.!?]+[\"')\]]*)(\s+|$)")
_CLAUSE_END = re.compile(r"([,;:—])\s+")
_ABBREV = re.compile(r"\b(Mr|Mrs|Ms|Dr|St|vs|e\.g|i\.e|etc|No)\.$", re.I)
_MARKUP = re.compile(r"[*_#`>|~]+|\[[^\]]*\]\([^)]*\)")


def speakable(text: str) -> str:
    """Strip markdown and other things a voice should not read out."""
    text = _MARKUP.sub("", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class PhraseChunker:
    def __init__(self, first_min_words: int = 4, max_words: int = 40) -> None:
        self.first_min_words, self.max_words = first_min_words, max_words
        self._buf = ""
        self._emitted = 0

    def feed(self, piece: str) -> list[str]:
        self._buf += piece
        out = []
        while True:
            cut = self._find_cut()
            if cut is None:
                break
            chunk, self._buf = speakable(self._buf[:cut]), self._buf[cut:].lstrip()
            if chunk and re.search(r"[A-Za-z0-9]", chunk):
                out.append(chunk)
                self._emitted += 1
        return out

    def _find_cut(self) -> int | None:
        for m in _SENT_END.finditer(self._buf):
            head = self._buf[:m.end(1)]
            if _ABBREV.search(head) or re.search(r"\d\.$", head) and m.group(2) == "":
                continue                      # "Dr." / a number still being written ("3." of "3.5")
            if m.group(2) == "" and not head.rstrip().endswith(("!", "?")):
                continue                      # wait for the space after a period to be sure
            return m.end()
        if self._emitted == 0:
            for m in _CLAUSE_END.finditer(self._buf):
                if len(self._buf[:m.start()].split()) >= self.first_min_words:
                    return m.end()
        words = self._buf.split()
        if len(words) > self.max_words:       # run-on text: cut at the last space
            return self._buf.rfind(" ") + 1
        return None

    def flush(self) -> list[str]:
        chunk, self._buf = speakable(self._buf), ""
        return [chunk] if chunk and re.search(r"[A-Za-z0-9]", chunk) else []
