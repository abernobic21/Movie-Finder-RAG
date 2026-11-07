import pytest
from fastapi.testclient import TestClient
from qdrant_client.http.exceptions import ApiException

from movie_finder.api.main import app
from movie_finder.models import Chunk, MovieResult, ScoredChunk


class StubPipeline:
    def __init__(self, error=None):
        self.error = error
        self.requested = None

    def search(self, query, top_k=5):
        self.requested = (query, top_k)
        if self.error:
            raise self.error
        chunk = Chunk(chunk_id="jaws_1975#0", movie_id="jaws_1975", title="Jaws", year=1975, text="A shark attacks.", position=0)
        return [MovieResult(movie_id="jaws_1975", title="Jaws", year=1975, score=0.03, passages=(ScoredChunk(chunk, 0.03),))]


class StubQdrant:
    def __init__(self, error=None):
        self.error = error

    def get_collections(self):
        if self.error:
            raise self.error


@pytest.fixture
def client():
    app.state.pipeline = StubPipeline()
    app.state.client = StubQdrant()
    return TestClient(app)


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_returns_503_when_qdrant_is_down(client):
    app.state.client = StubQdrant(error=ApiException("down"))
    assert client.get("/health").status_code == 503


def test_search_returns_movies_with_passages(client):
    response = client.post("/search", json={"query": "shark movie"})

    assert response.status_code == 200
    assert response.json() == {
        "results": [
            {
                "movie_id": "jaws_1975",
                "title": "Jaws",
                "year": 1975,
                "score": 0.03,
                "passages": [{"text": "A shark attacks.", "score": 0.03}],
            }
        ]
    }


def test_search_strips_query_and_passes_top_k(client):
    client.post("/search", json={"query": "  shark movie  ", "top_k": 3})
    assert app.state.pipeline.requested == ("shark movie", 3)


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"query": "   "},
        {"query": "x" * 1001},
        {"query": "shark", "top_k": 0},
        {"query": "shark", "top_k": 21},
    ],
)
def test_search_rejects_invalid_input(client, body):
    assert client.post("/search", json=body).status_code == 422


def test_search_returns_503_when_qdrant_is_down(client):
    app.state.pipeline = StubPipeline(error=ApiException("down"))
    assert client.post("/search", json={"query": "shark"}).status_code == 503
