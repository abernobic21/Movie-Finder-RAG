from typing import Protocol

from pydantic import BaseModel


class LLMError(Exception):
    """The LLM is not configured, the request failed, or the output is unusable."""


class LLM(Protocol):
    def generate(self, system: str, user: str, schema: type[BaseModel]) -> str:
        """Return JSON text that should match `schema`. The caller still validates it."""
        ...
