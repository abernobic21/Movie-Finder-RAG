import re

import numpy as np
from rank_bm25 import BM25Okapi

from movie_finder.models import Chunk, ScoredChunk


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


class BM25Retriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.index = BM25Okapi([tokenize(chunk.text) for chunk in chunks])

    def search(self, query: str, top_k: int) -> list[ScoredChunk]:
        scores = self.index.get_scores(tokenize(query))
        best = np.argsort(scores)[::-1][:top_k]
        return [ScoredChunk(self.chunks[i], float(scores[i])) for i in best if scores[i] > 0]
