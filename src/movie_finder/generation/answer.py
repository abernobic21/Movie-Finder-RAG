from pydantic import BaseModel, ValidationError

from movie_finder.generation.base import LLM, LLMError
from movie_finder.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from movie_finder.models import Answer, MovieResult
from dataclasses import replace



class LLMChoice(BaseModel):
    candidate: int | None
    passage: int | None
    explanation: str


def answer(query: str, candidates: list[MovieResult], llm: LLM, passages_per_movie: int = 2) -> Answer:
    if not candidates:
        return Answer(match=None, passage=None, explanation="No candidate movies were found.", candidates=())

    shown = [replace(movie, passages=movie.passages[:passages_per_movie]) for movie in candidates]

    text = llm.generate(SYSTEM_PROMPT, build_user_prompt(query, shown), LLMChoice)
    try:
        choice = LLMChoice.model_validate_json(text)
    except ValidationError as e:
        raise LLMError(f"LLM output does not match the schema: {e}") from e

    if choice.candidate is None:
        return Answer(match=None, passage=None, explanation=choice.explanation, candidates=tuple(shown))
    if not 1 <= choice.candidate <= len(shown):
        raise LLMError(f"LLM chose candidate {choice.candidate}, but there are {len(shown)}")

    match = shown[choice.candidate - 1]
    if choice.passage is None or not 1 <= choice.passage <= len(match.passages):
        raise LLMError(f"LLM chose passage {choice.passage} of {match.title}, which was not shown")

    return Answer(
        match=match,
        passage=match.passages[choice.passage - 1].chunk.text,
        explanation=choice.explanation,
        candidates=tuple(shown),
    )

