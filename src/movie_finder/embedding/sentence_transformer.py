import re


class SentenceTransformerEmbedder:
    def __init__(self, model: str, query_prefix: str = "", document_prefix: str = "", batch_size: int = 64):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model)
        self._name = re.sub(r"\W+", "_", model.rsplit("/", 1)[-1].casefold())
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.batch_size = batch_size

    @property
    def name(self) -> str:
        return self._name

    @property
    def dimension(self) -> int:
        return self._model.get_embedding_dimension()

    @property
    def max_tokens(self) -> int:
        return self._model.max_seq_length - self.tokenizer.num_special_tokens_to_add()

    @property
    def tokenizer(self):
        return self._model.tokenizer

    def _encode(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode([self.document_prefix + text for text in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self.query_prefix + text])[0]
