import pytest

from movie_finder.models import Chunk, ScoredChunk
from movie_finder.retrieval.bm25 import BM25Retriever
from movie_finder.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion


def chunk(chunk_id, text="text", movie_id=None):
    movie_id = movie_id or chunk_id.split("#")[0]
    return Chunk(chunk_id=chunk_id, movie_id=movie_id, title=movie_id, year=2000, text=text, position=0)


def scored(chunk_id, score=1.0):
    return ScoredChunk(chunk(chunk_id), score)


def ids(results):
    return [r.chunk.chunk_id for r in results]


def test_rrf_calculation_by_hand():
    results = reciprocal_rank_fusion([[scored("a#0"), scored("b#0")], [scored("b#0")]], k=60)

    assert ids(results) == ["b#0", "a#0"]
    assert results[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert results[1].score == pytest.approx(1 / 61)


def test_rrf_ignores_score_scale():
    results = reciprocal_rank_fusion([[scored("a#0", 1000.0)], [scored("b#0", 0.001)]])
    assert results[0].score == results[1].score


def test_rrf_handles_an_empty_list():
    assert ids(reciprocal_rank_fusion([[], [scored("a#0")]])) == ["a#0"]


class StubRetriever:
    def __init__(self, results):
        self.results = results
        self.requested = None

    def search(self, query, top_k):
        self.requested = top_k
        return self.results


def test_hybrid_merges_and_limits():
    first = StubRetriever([scored("a#0"), scored("b#0")])
    second = StubRetriever([scored("b#0"), scored("c#0")])

    results = HybridRetriever([first, second]).search("query", top_k=2)

    assert ids(results)[0] == "b#0"
    assert len(results) == 2
    assert first.requested == second.requested == 2


def test_bm25_finds_matching_chunk_and_ignores_case():
    chunks = [chunk("a#0", "a shark attacks swimmers"), chunk("b#0", "two friends open a bakery"), chunk("c#0", "a robot learns to paint")]
    assert ids(BM25Retriever(chunks).search("SHARK", top_k=5)) == ["a#0"]


def test_bm25_returns_nothing_without_a_match():
    chunks = [chunk("a#0", "a shark attacks"), chunk("b#0", "friends open a bakery"), chunk("c#0", "a robot paints")]
    assert BM25Retriever(chunks).search("spaceship", top_k=5) == []
