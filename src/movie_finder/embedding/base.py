from typing import Any, Protocol


class Embedder(Protocol):
    @property
    def name(self) -> str:
        """Identifies the model in the collection name, e.g. all_minilm_l6_v2."""
        ...

    @property
    def dimension(self) -> int: ...

    @property
    def max_tokens(self) -> int:
        """The longest input the model reads, excluding special tokens. Longer text is cut off."""
        ...

    @property
    def tokenizer(self) -> Any: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...
