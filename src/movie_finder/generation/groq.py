import json
import os

from groq import APIError, Groq
from pydantic import BaseModel

from movie_finder.generation.base import LLMError


class GroqLLM:
    def __init__(self, model: str, temperature: float = 0.0):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise LLMError("GROQ_API_KEY is not set")
        self._client = Groq(api_key=api_key)
        self.model = model
        self.temperature = temperature

    def generate(self, system: str, user: str, schema: type[BaseModel]) -> str:
        schema_text = json.dumps(schema.model_json_schema())
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": f"{system}\n\nReply only with JSON matching this schema:\n{schema_text}"},
                    {"role": "user", "content": user},
                ],
            )
        except APIError as e:
            raise LLMError(f"Groq request failed: {e}") from e
        content = response.choices[0].message.content
        if not content:
            raise LLMError("Groq returned an empty response")
        return content
