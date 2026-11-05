from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    movie_id: str
    title: str
    year: int
    plot: str
    genre: str | None = None
    origin: str | None = None

