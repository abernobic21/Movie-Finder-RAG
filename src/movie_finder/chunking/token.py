from movie_finder.models import Chunk, Document


class TokenChunker:
    """Fixed-size overlapping windows, counted with the embedding model's own tokenizer."""

    def __init__(self, tokenizer, chunk_size: int = 200, overlap: int = 40):
        if chunk_size <= 0 or not 0 <= overlap < chunk_size:
            raise ValueError(f"Invalid chunk_size={chunk_size}, overlap={overlap}")
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.overlap = overlap

    @property
    def name(self) -> str:
        return f"token{self.chunk_size}_o{self.overlap}"

    @property
    def max_tokens(self) -> int:
        return self.chunk_size

    def chunk(self, doc: Document) -> list[Chunk]:
        offsets = self.tokenizer(
            doc.plot, add_special_tokens=False, return_offsets_mapping=True, verbose=False
        )["offset_mapping"]

        chunks = []
        step = self.chunk_size - self.overlap
        for start in range(0, len(offsets), step):
            window = offsets[start : start + self.chunk_size]
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.movie_id}#{len(chunks)}",
                    movie_id=doc.movie_id,
                    title=doc.title,
                    year=doc.year,
                    text=doc.plot[window[0][0] : window[-1][1]],
                    position=len(chunks),
                )
            )
            if start + self.chunk_size >= len(offsets):
                break
        return chunks
