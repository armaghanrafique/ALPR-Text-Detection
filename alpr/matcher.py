"""
Precise plate-text matching against a watchlist of known plates.

Two-stage matching:
  1. Normalise the OCR text and the watchlist (uppercase, strip non-alnum,
     and undo the most common OCR character confusions) and try an EXACT match.
  2. If no exact match, fall back to fuzzy matching (rapidfuzz) and accept the
     best candidate only if it clears `fuzzy_threshold`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Characters that OCR routinely swaps. We canonicalise letters that look like
# digits so "O" and "0" (etc.) compare equal.
_CONFUSION = str.maketrans({"O": "0", "I": "1", "Q": "0", "Z": "2", "S": "5", "B": "8"})


def normalize(text: str) -> str:
    """Uppercase, keep only A-Z/0-9, then fold look-alike characters."""
    cleaned = "".join(ch for ch in text.upper() if ch.isalnum())
    return cleaned.translate(_CONFUSION)


@dataclass
class MatchResult:
    matched: bool
    plate: str | None      # the canonical watchlist entry that matched
    score: float           # 0..100
    exact: bool


class PlateMatcher:
    def __init__(self, watchlist_path: str | None, fuzzy_threshold: float = 85.0) -> None:
        self.fuzzy_threshold = fuzzy_threshold
        self.entries: list[str] = []          # original, human-readable
        self._norm_to_entry: dict[str, str] = {}

        if watchlist_path and os.path.exists(watchlist_path):
            with open(watchlist_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    self.entries.append(line)
                    self._norm_to_entry[normalize(line)] = line

        # rapidfuzz is optional; degrade gracefully to difflib if missing.
        try:
            from rapidfuzz import fuzz, process

            self._fuzz = fuzz
            self._process = process
        except Exception:  # pragma: no cover
            self._fuzz = None
            self._process = None

    @property
    def loaded(self) -> bool:
        return bool(self.entries)

    def match(self, text: str) -> MatchResult:
        if not text or not self.entries:
            return MatchResult(matched=False, plate=None, score=0.0, exact=False)

        norm = normalize(text)

        # 1) Exact match on normalised forms.
        if norm in self._norm_to_entry:
            return MatchResult(
                matched=True, plate=self._norm_to_entry[norm], score=100.0, exact=True
            )

        # 2) Fuzzy fallback.
        norms = list(self._norm_to_entry.keys())
        if self._process is not None:
            best = self._process.extractOne(norm, norms, scorer=self._fuzz.ratio)
            if best is not None:
                cand_norm, score, _ = best
            else:
                cand_norm, score = None, 0.0
        else:
            import difflib

            cand_norm, score = None, 0.0
            for cand in norms:
                s = difflib.SequenceMatcher(None, norm, cand).ratio() * 100
                if s > score:
                    cand_norm, score = cand, s

        if cand_norm is not None and score >= self.fuzzy_threshold:
            return MatchResult(
                matched=True,
                plate=self._norm_to_entry[cand_norm],
                score=float(score),
                exact=False,
            )

        return MatchResult(matched=False, plate=None, score=float(score), exact=False)
