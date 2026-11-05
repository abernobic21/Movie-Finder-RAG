import pytest
from qdrant_client import QdrantClient

from movie_finder.config import ComponentConfig, PipelineConfig, load_config
from movie_finder.data import write_jsonl
from movie_finder.models import Document
from movie_finder.pipeline import build_index, build_pipeline

MOVIES = [
    Document("shark_1975", "Shark", 1975, "a great white shark attacks swimmers near a quiet beach town"),
    Document("bakery_1999", "Bakery", 1999, "two friends open a small bakery and compete for customers"),
    Document("robot_2010", "Robot", 2010, "a lonely robot learns to paint after its factory closes"),
]


@pytest.fixture
def client():
    return QdrantClient(":memory:")


@pytest.fixture
def make_config(tmp_path, fake_embedder):
    data_path = tmp_path / "movies.jsonl"
    write_jsonl(MOVIES, data_path)

    def make(retriever="hybrid", chunk_size=5, overlap=1, max_tokens=100):
        return PipelineConfig(
            name="test",
            data_path=data_path,
            index_root=tmp_path / "indexes",
            chunker=ComponentConfig(type="token", chunk_size=chunk_size, overlap=overlap),
            embedder=ComponentConfig(type="fake", max_tokens=max_tokens),
            retriever=ComponentConfig(type=retriever),
        )

    return make


@pytest.mark.parametrize("retriever", ["vector", "bm25", "hybrid"])
def test_each_retriever_finds_the_right_movie(client, make_config, retriever):
    config = make_config(retriever=retriever)
    build_index(config, client)

    results = build_pipeline(config, client).search("shark attacks swimmers", top_k=3)

    assert results[0].movie_id == "shark_1975"
    assert len({r.movie_id for r in results}) == len(results)


def test_changing_settings_creates_a_separate_collection(client, make_config):
    first, _ = build_index(make_config(chunk_size=5), client)
    second, _ = build_index(make_config(chunk_size=8), client)

    assert first == "token5_o1__fake"
    assert second == "token8_o1__fake"
    assert client.collection_exists(first) and client.collection_exists(second)


def test_existing_index_is_reused_unless_rebuild_is_requested(client, make_config):
    config = make_config()

    assert build_index(config, client)[1] is True
    assert build_index(config, client)[1] is False
    assert build_index(config, client, rebuild=True)[1] is True


def test_retrievers_share_one_index(client, make_config):
    built = build_index(make_config(retriever="hybrid"), client)[1]
    assert built is True
    assert build_index(make_config(retriever="bm25"), client)[1] is False


def test_incomplete_index_is_rebuilt(client, make_config):
    config = make_config()
    collection, _ = build_index(config, client)
    (config.index_root / collection / "chunks.jsonl").unlink()

    assert build_index(config, client)[1] is True


def test_searching_without_an_index_fails_clearly(client, make_config):
    with pytest.raises(RuntimeError, match="build_index"):
        build_pipeline(make_config(), client)


def test_chunks_longer_than_the_model_limit_are_rejected(client, make_config):
    with pytest.raises(ValueError, match="cut off"):
        build_index(make_config(chunk_size=50, overlap=0, max_tokens=20), client)


def test_unknown_component_lists_the_options(client, make_config):
    config = make_config()
    build_index(config, client)
    config.retriever = ComponentConfig(type="magic")

    with pytest.raises(ValueError, match="hybrid"):
        build_pipeline(config, client)


def test_config_passes_extra_fields_as_params(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "name: x\ndata_path: d.jsonl\n"
        "chunker: {type: token, chunk_size: 10}\n"
        "embedder: {type: fake}\n"
        "retriever: {type: hybrid, rrf_k: 30}\n",
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.chunker.params == {"chunk_size": 10}
    assert config.retriever.params == {"rrf_k": 30}


def test_config_rejects_unknown_top_level_fields(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "name: x\ndata_path: d.jsonl\ntypo: 1\n"
        "chunker: {type: token}\nembedder: {type: fake}\nretriever: {type: bm25}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_config(path)
