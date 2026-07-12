"""Word error rate (WER) -- score the engine's STT against ground-truth text.

WER = (substitutions + insertions + deletions) / reference_words, the standard
ASR accuracy metric. Used to turn 'STT works' into a number when the injected
clips carry known transcripts (real recordings, or the synth script).
"""
from __future__ import annotations

import re
from dataclasses import dataclass


def _norm(s: str) -> list[str]:
    """Lowercase, strip punctuation, split to words (so 'Paris.' == 'paris')."""
    return re.sub(r"[^\w\s]", " ", (s or "").lower()).split()


def wer(reference: str, hypothesis: str) -> float:
    """Levenshtein word distance / reference length. 0.0 = perfect; >=1.0 = bad."""
    r, h = _norm(reference), _norm(hypothesis)
    if not r:
        return 0.0 if not h else 1.0
    # classic DP edit distance over word lists
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i]
        for j, hw in enumerate(h, 1):
            cur.append(min(
                prev[j] + 1,            # deletion
                cur[j - 1] + 1,         # insertion
                prev[j - 1] + (rw != hw),  # substitution / match
            ))
        prev = cur
    return prev[-1] / len(r)


@dataclass
class SttScore:
    pairs: list[tuple[str, str, float]]  # (reference, hypothesis, wer)
    mean_wer: float
    n: int  # labelled references that received a non-blank hypothesis


def score_transcripts(references: list[str], hypotheses: list[str]) -> SttScore:
    """Pair each reference with the best-matching hypothesis (greedy, order-
    preserving) and aggregate WER. Hypotheses are the engine's recognized user
    finals; references are the injected clips' ground truth. Robust to extra
    spurious finals (self-echo, fragments) by taking the min-WER hypothesis."""
    pairs: list[tuple[str, str, float]] = []
    pool = [hypothesis for hypothesis in hypotheses if hypothesis.strip()]
    matched = 0
    for ref in references:
        if not ref:
            continue
        if not pool:
            pairs.append((ref, "", 1.0))
            continue
        # best hypothesis for this reference, then consume it
        best_i, best_w = min(
            ((i, wer(ref, h)) for i, h in enumerate(pool)), key=lambda t: t[1]
        )
        pairs.append((ref, pool.pop(best_i), best_w))
        matched += 1
    mean = sum(w for _, _, w in pairs) / len(pairs) if pairs else 0.0
    return SttScore(pairs=pairs, mean_wer=round(mean, 3), n=matched)
