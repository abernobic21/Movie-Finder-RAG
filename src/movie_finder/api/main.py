import os
import logging

from contextlib import asynccontextmanager
from pathlib import Path
from movie_finder.api.schemas import AskRequest, AskResponse, MatchResponse, MovieResponse, PassageResponse, SearchRequest, SearchResponse


from fastapi import FastAPI, HTTPException, Request
from movie_finder.models import MovieResult
from qdrant_client.http.exceptions import ApiException

from movie_finder.config import load_config
from movie_finder.pipeline import build_pipeline, get_client

from movie_finder.generation.answer import answer
from movie_finder.generation.base import LLMError
from movie_finder.registry import build_llm

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config(Path(os.environ.get("CONFIG_PATH", "configs/baseline.yaml")))
    app.state.client = get_client()
    app.state.pipeline = build_pipeline(config, app.state.client)
    app.state.llm = None
    if config.llm is not None:
        try:
            app.state.llm = build_llm(config.llm)
        except LLMError as e:
            logger.warning("/ask is disabled: %s", e)
    yield


app = FastAPI(title="Movie Finder", lifespan=lifespan)


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    try:
        request.app.state.client.get_collections()
    except ApiException:
        raise HTTPException(status_code=503, detail="Qdrant is unreachable")
    return {"status": "ok"}

@app.post("/search")
def search(body: SearchRequest, request: Request) -> SearchResponse:
    try:
        movies = request.app.state.pipeline.search(body.query, top_k=body.top_k)
    except ApiException:
        raise HTTPException(status_code=503, detail="Qdrant is unreachable")

    return SearchResponse(results=[to_movie_response(movie) for movie in movies])

@app.post("/ask")
def ask(body: AskRequest, request: Request) -> AskResponse:
    llm = request.app.state.llm
    if llm is None:
        raise HTTPException(status_code=503, detail="LLM is not configured")

    try:
        candidates = request.app.state.pipeline.search(body.query, top_k=body.top_k)
    except ApiException:
        raise HTTPException(status_code=503, detail="Qdrant is unreachable")

    try:
        result = answer(body.query, candidates, llm)
    except LLMError as e:
        logger.warning("LLM failed: %s", e)
        raise HTTPException(status_code=502, detail="The LLM request failed")

    return AskResponse(
        match=None if result.match is None else MatchResponse(
            movie_id=result.match.movie_id, title=result.match.title, year=result.match.year
        ),
        explanation=result.explanation,
        passage=result.passage,
        candidates=[to_movie_response(movie) for movie in result.candidates],
    )


def to_movie_response(movie: MovieResult) -> MovieResponse:
    return MovieResponse(
        movie_id=movie.movie_id,
        title=movie.title,
        year=movie.year,
        score=movie.score,
        passages=[PassageResponse(text=p.chunk.text, score=p.score) for p in movie.passages],
    )


