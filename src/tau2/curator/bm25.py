"""Minimal, dependency-free BM25Okapi.

``rank-bm25`` is only a tau2 *knowledge* extra (not a core dep), and the shared eval env must
not be mutated, so we vendor the ~40-line Okapi BM25 here. API-compatible subset of
``rank_bm25.BM25Okapi``: construct with a tokenized corpus (``List[List[str]]``), then call
``get_scores(query_tokens) -> List[float]``. Same k1/b defaults (1.5 / 0.75) and same scoring
formula, so retrieval ranking matches the reference implementation.
"""

from __future__ import annotations

import math
from typing import List


class BM25Okapi:
    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75,
                 epsilon: float = 0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.corpus_size = len(corpus)
        self.doc_len = [len(doc) for doc in corpus]
        self.avgdl = (sum(self.doc_len) / self.corpus_size) if self.corpus_size else 0.0

        # term frequency per doc + document frequency per term
        self.doc_freqs: List[dict] = []
        df: dict = {}
        for doc in corpus:
            freqs: dict = {}
            for term in doc:
                freqs[term] = freqs.get(term, 0) + 1
            self.doc_freqs.append(freqs)
            for term in freqs:
                df[term] = df.get(term, 0) + 1

        # idf with the standard rank_bm25 negative-idf flooring (avg * epsilon).
        self.idf: dict = {}
        idf_sum = 0.0
        negatives = []
        for term, freq in df.items():
            idf = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5))
            self.idf[term] = idf
            idf_sum += idf
            if idf < 0:
                negatives.append(term)
        avg_idf = (idf_sum / len(self.idf)) if self.idf else 0.0
        floor = self.epsilon * avg_idf
        for term in negatives:
            self.idf[term] = floor

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        scores = [0.0] * self.corpus_size
        for term in query_tokens:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i in range(self.corpus_size):
                freq = self.doc_freqs[i].get(term, 0)
                if not freq:
                    continue
                denom = freq + self.k1 * (
                    1 - self.b + self.b * self.doc_len[i] / (self.avgdl or 1.0)
                )
                scores[i] += idf * (freq * (self.k1 + 1) / denom)
        return scores
