from movie_finder.models import ScoredChunk
from movie_finder.retrieval.base import Retriever


def reciprocal_rank_fusion(ranked_lists: list[list[ScoredChunk]], k: int = 60) -> list[ScoredChunk]:
    """Combines rankings by position, since BM25 and cosine scores aren't on the same scale."""
    scores: dict[str, float] = {}
    chunks = {}
    for ranked in ranked_lists:
        for rank, scored in enumerate(ranked, start=1):
            chunk_id = scored.chunk.chunk_id
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (k + rank)
            chunks[chunk_id] = scored.chunk
    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
    return [ScoredChunk(chunks[chunk_id], scores[chunk_id]) for chunk_id in ordered]


class HybridRetriever:
    def __init__(self, retrievers: list[Retriever], rrf_k: int = 60):
        self.retrievers = retrievers
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int) -> list[ScoredChunk]:
        ranked_lists = [retriever.search(query, top_k) for retriever in self.retrievers]
        return reciprocal_rank_fusion(ranked_lists, k=self.rrf_k)[:top_k]
