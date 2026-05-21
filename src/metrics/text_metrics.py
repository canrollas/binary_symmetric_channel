"""Text-level error metrics: CER, WER, BLEU, edit distance."""

from typing import List
import numpy as np


def edit_distance(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def character_error_rate(original: str, recovered: str) -> float:
    if len(original) == 0:
        return 0.0 if len(recovered) == 0 else 1.0
    return edit_distance(original, recovered) / len(original)


def word_error_rate(original: str, recovered: str) -> float:
    orig_words = original.split()
    recv_words = recovered.split()
    if len(orig_words) == 0:
        return 0.0 if len(recv_words) == 0 else 1.0
    return edit_distance(" ".join(orig_words), " ".join(recv_words)) / len(" ".join(orig_words))


def word_error_rate_words(original: str, recovered: str) -> float:
    orig_words = original.split()
    recv_words = recovered.split()
    if len(orig_words) == 0:
        return 0.0 if len(recv_words) == 0 else 1.0
    m, n = len(orig_words), len(recv_words)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if orig_words[i-1] == recv_words[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n] / m


def normalized_edit_distance(a: str, b: str) -> float:
    if len(a) == 0 and len(b) == 0:
        return 0.0
    return edit_distance(a, b) / max(len(a), len(b))


def _ngrams(tokens: List[str], n: int) -> List[tuple]:
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def bleu_score(reference: str, hypothesis: str, max_n: int = 4) -> float:
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    if len(hyp_tokens) == 0:
        return 0.0

    brevity = min(1.0, len(hyp_tokens) / max(len(ref_tokens), 1))

    log_avg = 0.0
    effective_n = 0
    for n in range(1, max_n + 1):
        ref_ngrams = _ngrams(ref_tokens, n)
        hyp_ngrams = _ngrams(hyp_tokens, n)
        if not hyp_ngrams:
            continue  # skip n-gram order if too short
        ref_counts: dict = {}
        for g in ref_ngrams:
            ref_counts[g] = ref_counts.get(g, 0) + 1
        clipped = 0
        for g in hyp_ngrams:
            if ref_counts.get(g, 0) > 0:
                clipped += 1
                ref_counts[g] -= 1
        precision = clipped / len(hyp_ngrams)
        if precision == 0:
            return 0.0
        log_avg += np.log(precision)
        effective_n += 1

    if effective_n == 0:
        return 0.0
    return brevity * np.exp(log_avg / effective_n)
