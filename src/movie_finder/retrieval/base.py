from typing import Protocol

from movie_finder.models import ScoredChunk


class Retriever(Protocol):
    def search(self, query: str, top_k: int) -> list[ScoredChunk]:
        """Returns chunks ordered best first. Scores are only comparable within one retriever."""
        ...
