import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from qdrant_client.http.exceptions import ApiException

from movie_finder.config import load_config
from movie_finder.pipeline import build_pipeline, get_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config(Path(os.environ.get("CONFIG_PATH", "configs/baseline.yaml")))
    app.state.client = get_client()
    app.state.pipeline = build_pipeline(config, app.state.client)
    yield


app = FastAPI(title="Movie Finder", lifespan=lifespan)


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    try:
        request.app.state.client.get_collections()
    except ApiException:
        raise HTTPException(status_code=503, detail="Qdrant is unreachable")
    return {"status": "ok"}
