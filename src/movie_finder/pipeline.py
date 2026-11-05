import json
import os
import uuid
from dataclasses import asdict
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from movie_finder.chunking.base import Chunker
from movie_finder.config import PipelineConfig
from movie_finder.data import read_jsonl
from movie_finder.embedding.base import Embedder
from movie_finder.models import Chunk, MovieResult, ScoredChunk
from movie_finder.registry import RetrieverResources, build_chunker, build_embedder, build_retriever
from movie_finder.retrieval.base import Retriever

UPSERT_BATCH_SIZE = 256


def get_client() -> QdrantClient:
    return QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))


def collection_name(chunker: Chunker, embedder: Embedder) -> str:
    return f"{chunker.name}__{embedder.name}"


def _build_components(config: PipelineConfig) -> tuple[Embedder, Chunker]:
    embedder = build_embedder(config.embedder)
    chunker = build_chunker(config.chunker, embedder)
    if chunker.max_tokens > embedder.max_tokens:
        raise ValueError(
            f"Chunks of {chunker.max_tokens} tokens would be cut off: "
            f"{embedder.name} reads at most {embedder.max_tokens}"
        )
    return embedder, chunker


def _chunks_path(config: PipelineConfig, collection: str) -> Path:
    return config.index_root / collection / "chunks.jsonl"


def _read_chunks(path: Path) -> list[Chunk]:
    with path.open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f]


def _write_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def _index_is_complete(client: QdrantClient, collection: str, chunks_path: Path) -> bool:
    if not chunks_path.exists() or not client.collection_exists(collection):
        return False
    expected = len(_read_chunks(chunks_path))
    return client.count(collection, exact=True).count == expected


def build_index(config: PipelineConfig, client: QdrantClient, rebuild: bool = False) -> tuple[str, bool]:
    """Chunks and embeds the data. Returns the collection name and whether anything was built."""
    embedder, chunker = _build_components(config)
    collection = collection_name(chunker, embedder)
    chunks_path = _chunks_path(config, collection)

    if not rebuild and _index_is_complete(client, collection, chunks_path):
        return collection, False

    chunks = [chunk for doc in read_jsonl(config.data_path) for chunk in chunker.chunk(doc)]
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection,
        vectors_config=VectorParams(size=embedder.dimension, distance=Distance.COSINE),
    )

    for start in range(0, len(chunks), UPSERT_BATCH_SIZE):
        batch = chunks[start : start + UPSERT_BATCH_SIZE]
        vectors = embedder.embed_documents([chunk.text for chunk in batch])
        client.upsert(
            collection,
            points=[
                # Qdrant IDs must be integers or UUIDs, so derive a stable UUID from the chunk ID
                PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id)), vector=vector, payload=asdict(chunk))
                for chunk, vector in zip(batch, vectors)
            ],
        )
        print(f"Indexed {start + len(batch)}/{len(chunks)} chunks", flush=True)

    # Written last, so an interrupted build isn't mistaken for a finished one
    _write_chunks(chunks, chunks_path)
    return collection, True


def group_by_movie(scored_chunks: list[ScoredChunk], top_k: int) -> list[MovieResult]:
    """Ranks each movie by its best chunk, so one movie can't fill the results."""
    grouped: dict[str, list[ScoredChunk]] = {}
    for scored in scored_chunks:
        grouped.setdefault(scored.chunk.movie_id, []).append(scored)

    results = []
    for movie_id, passages in grouped.items():
        passages = sorted(passages, key=lambda s: -s.score)
        best = passages[0]
        results.append(MovieResult(movie_id, best.chunk.title, best.chunk.year, best.score, tuple(passages)))
    results.sort(key=lambda r: (-r.score, r.movie_id))
    return results[:top_k]


class Pipeline:
    def __init__(self, retriever: Retriever, candidate_chunks: int):
        self.retriever = retriever
        self.candidate_chunks = candidate_chunks

    def search(self, query: str, top_k: int = 5) -> list[MovieResult]:
        return group_by_movie(self.retriever.search(query, self.candidate_chunks), top_k)


def build_pipeline(config: PipelineConfig, client: QdrantClient) -> Pipeline:
    embedder, chunker = _build_components(config)
    collection = collection_name(chunker, embedder)
    chunks_path = _chunks_path(config, collection)
    if not _index_is_complete(client, collection, chunks_path):
        raise RuntimeError(f"Index '{collection}' isn't built. Run scripts/build_index.py first.")

    resources = RetrieverResources(client, collection, embedder, _read_chunks(chunks_path))
    return Pipeline(build_retriever(config.retriever, resources), config.candidate_chunks)
