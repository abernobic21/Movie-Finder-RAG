import hashlib
import re

import pytest

from movie_finder.registry import EMBEDDERS


class WhitespaceTokenizer:
    """Each word is one token, with the same output format as a Hugging Face tokenizer."""

    def __call__(self, text, **kwargs):
        return {"offset_mapping": [m.span() for m in re.finditer(r"\S+", text)]}


class FakeEmbedder:
    """Bag-of-words vectors: fast, offline, and texts sharing words are similar."""

    name = "fake"
    dimension = 64
    tokenizer = WhitespaceTokenizer()

    def __init__(self, max_tokens: int = 100):
        self.max_tokens = max_tokens

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for word in re.findall(r"\w+", text.casefold()):
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dimension] += 1.0
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

class FakeLLM:
    """Returns a fixed reply and records the prompt it was given."""

    def __init__(self, reply: str):
        self.reply = reply
        self.user = None

    def generate(self, system, user, schema):
        self.user = user
        return self.reply


@pytest.fixture
def fake_llm():
    return FakeLLM


@pytest.fixture
def fake_embedder(monkeypatch):
    monkeypatch.setitem(EMBEDDERS, "fake", FakeEmbedder)
    return FakeEmbedder
