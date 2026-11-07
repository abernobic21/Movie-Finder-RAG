from dataclasses import dataclass

from qdrant_client import QdrantClient

from movie_finder.chunking.base import Chunker
from movie_finder.chunking.token import TokenChunker
from movie_finder.config import ComponentConfig
from movie_finder.embedding.base import Embedder
from movie_finder.embedding.sentence_transformer import SentenceTransformerEmbedder
from movie_finder.generation.base import LLM
from movie_finder.generation.groq import GroqLLM
from movie_finder.models import Chunk
from movie_finder.retrieval.base import Retriever
from movie_finder.retrieval.bm25 import BM25Retriever
from movie_finder.retrieval.hybrid import HybridRetriever
from movie_finder.retrieval.vector import VectorRetriever
from movie_finder.generation.groq import GroqLLM


@dataclass(frozen=True)
class RetrieverResources:
    client: QdrantClient
    collection: str
    embedder: Embedder
    chunks: list[Chunk]


def _vector(res: RetrieverResources) -> Retriever:
    return VectorRetriever(res.client, res.collection, res.embedder)


def _bm25(res: RetrieverResources) -> Retriever:
    return BM25Retriever(res.chunks)


def _hybrid(res: RetrieverResources, rrf_k: int = 60) -> Retriever:
    return HybridRetriever([_vector(res), _bm25(res)], rrf_k=rrf_k)


EMBEDDERS = {"sentence_transformer": SentenceTransformerEmbedder}
CHUNKERS = {"token": TokenChunker}
RETRIEVERS = {"vector": _vector, "bm25": _bm25, "hybrid": _hybrid}
LLMS = {"groq": GroqLLM}


def _lookup(options: dict, component: ComponentConfig, kind: str):
    if component.type not in options:
        raise ValueError(f"Unknown {kind} '{component.type}'. Options: {sorted(options)}")
    return options[component.type]


def build_embedder(config: ComponentConfig) -> Embedder:
    return _lookup(EMBEDDERS, config, "embedder")(**config.params)


def build_chunker(config: ComponentConfig, embedder: Embedder) -> Chunker:
    return _lookup(CHUNKERS, config, "chunker")(tokenizer=embedder.tokenizer, **config.params)


def build_retriever(config: ComponentConfig, resources: RetrieverResources) -> Retriever:
    return _lookup(RETRIEVERS, config, "retriever")(resources, **config.params)

def build_llm(config: ComponentConfig) -> LLM:
    return _lookup(LLMS, config, "llm")(**config.params)
