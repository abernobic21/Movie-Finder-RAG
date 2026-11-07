from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    movie_id: str
    title: str
    year: int
    plot: str
    genre: str | None = None
    origin: str | None = None


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    movie_id: str
    title: str
    year: int
    text: str
    position: int


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class MovieResult:
    movie_id: str
    title: str
    year: int
    score: float
    passages: tuple[ScoredChunk, ...]

@dataclass(frozen=True)
class Answer:
    match: MovieResult | None
    passage: str | None
    explanation: str
    candidates: tuple[MovieResult, ...]
