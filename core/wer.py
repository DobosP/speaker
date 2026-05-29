"""Word Error Rate — dependency-free, for STT regression measurement.

The STT redesign (docs/stt_redesign_2026-05.md) is gated on being able to MEASURE
accuracy instead of judging by a single sentence. This computes a normalized WER
(Levenshtein over lowercased, punctuation-stripped word tokens) plus a
substitution/insertion/deletion breakdown, so a model/resampler/decoder change
can be A/B'd on a replayed fixture and quoted as a delta.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_WORD = re.compile(r"[a-z0-9']+")


def normalize(text: str) -> list[str]:
    """Lowercase and split into comparable word tokens (drops punctuation)."""
    return _WORD.findall((text or "").lower())


@dataclass(frozen=True)
class WerResult:
    wer: float          # (sub + ins + del) / max(1, len(reference))
    substitutions: int
    insertions: int
    deletions: int
    ref_words: int
    hyp_words: int

    def as_dict(self) -> dict:
        return {
            "wer": round(self.wer, 4),
            "substitutions": self.substitutions,
            "insertions": self.insertions,
            "deletions": self.deletions,
            "ref_words": self.ref_words,
            "hyp_words": self.hyp_words,
        }


def word_error_rate(reference: str, hypothesis: str) -> WerResult:
    """Levenshtein WER with op breakdown. Empty reference -> 0.0 if hyp also
    empty, else 1.0 (all insertions)."""
    ref = normalize(reference)
    hyp = normalize(hypothesis)
    r, h = len(ref), len(hyp)
    if r == 0:
        return WerResult(0.0 if h == 0 else 1.0, 0, h, 0, 0, h)

    # DP cost matrix + op backtrace. cost[i][j] = edits to turn ref[:i] -> hyp[:j].
    cost = [[0] * (h + 1) for _ in range(r + 1)]
    for i in range(r + 1):
        cost[i][0] = i
    for j in range(h + 1):
        cost[0][j] = j
    for i in range(1, r + 1):
        for j in range(1, h + 1):
            if ref[i - 1] == hyp[j - 1]:
                cost[i][j] = cost[i - 1][j - 1]
            else:
                cost[i][j] = 1 + min(
                    cost[i - 1][j - 1],  # substitution
                    cost[i][j - 1],      # insertion
                    cost[i - 1][j],      # deletion
                )

    # Backtrace to count op types.
    i, j = r, h
    sub = ins = dele = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and cost[i][j] == cost[i - 1][j - 1]:
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and cost[i][j] == cost[i - 1][j - 1] + 1:
            sub += 1
            i, j = i - 1, j - 1
        elif j > 0 and cost[i][j] == cost[i][j - 1] + 1:
            ins += 1
            j -= 1
        else:
            dele += 1
            i -= 1
    return WerResult((sub + ins + dele) / r, sub, ins, dele, r, h)
