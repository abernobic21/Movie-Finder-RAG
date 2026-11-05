from qdrant_client import QdrantClient

from movie_finder.embedding.base import Embedder
from movie_finder.models import Chunk, ScoredChunk


class VectorRetriever:
    def __init__(self, client: QdrantClient, collection: str, embedder: Embedder):
        self.client = client
        self.collection = collection
        self.embedder = embedder

    def search(self, query: str, top_k: int) -> list[ScoredChunk]:
        hits = self.client.query_points(
            collection_name=self.collection,
            query=self.embedder.embed_query(query),
            limit=top_k,
            with_payload=True,
        ).points
        return [ScoredChunk(Chunk(**hit.payload), hit.score) for hit in hits]
