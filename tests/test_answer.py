import json

import pytest

from movie_finder.generation.answer import answer
from movie_finder.generation.base import LLMError
from movie_finder.models import Chunk, MovieResult, ScoredChunk


def movie(movie_id, *texts):
    passages = tuple(
        ScoredChunk(Chunk(f"{movie_id}#{i}", movie_id, movie_id, 2000, text, i), 1.0)
        for i, text in enumerate(texts)
    )
    return MovieResult(movie_id, movie_id, 2000, 1.0, passages)


CANDIDATES = [movie("jaws", "A shark attacks.", "The town hires a hunter.", "The boat sinks."), movie("up", "A house flies.")]


def reply(candidate, passage, explanation="because"):
    return json.dumps({"candidate": candidate, "passage": passage, "explanation": explanation})


def test_returns_the_chosen_movie_and_its_exact_passage(fake_llm):
    result = answer("shark movie", CANDIDATES, fake_llm(reply(1, 2)))

    assert result.match.movie_id == "jaws"
    assert result.passage == "The town hires a hunter."
    assert result.explanation == "because"
    assert len(result.candidates) == 2


def test_none_returns_no_match_but_keeps_candidates(fake_llm):
    result = answer("a dog in space", CANDIDATES, fake_llm(reply(None, None, "no dogs")))

    assert result.match is None
    assert result.passage is None
    assert len(result.candidates) == 2


def test_no_candidates_skips_the_llm(fake_llm):
    llm = fake_llm(reply(1, 1))
    result = answer("anything", [], llm)

    assert result.match is None
    assert llm.user is None


def test_prompt_shows_only_the_best_passages(fake_llm):
    llm = fake_llm(reply(1, 1))
    answer("shark movie", CANDIDATES, llm, passages_per_movie=2)

    assert "The town hires a hunter." in llm.user
    assert "The boat sinks." not in llm.user


@pytest.mark.parametrize(
    "bad_reply",
    [
        "not json",
        json.dumps({"candidate": 1}),
        reply(3, 1),
        reply(0, 1),
        reply(1, 3),
        reply(1, None),
    ],
)
def test_rejects_invalid_llm_output(fake_llm, bad_reply):
    with pytest.raises(LLMError):
        answer("shark movie", CANDIDATES, fake_llm(bad_reply))


def test_candidates_contain_only_the_passages_shown(fake_llm):
    result = answer("shark movie", CANDIDATES, fake_llm(reply(1, 1)), passages_per_movie=2)

    assert [len(movie.passages) for movie in result.candidates] == [2, 1]
