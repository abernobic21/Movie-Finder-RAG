from typing import Protocol

from movie_finder.models import Chunk, Document


class Chunker(Protocol):
    @property
    def name(self) -> str:
        """Identifies the settings in the collection name, e.g. token200_o40."""
        ...

    @property
    def max_tokens(self) -> int:
        """The longest chunk this chunker can produce."""
        ...

    def chunk(self, doc: Document) -> list[Chunk]: ...
