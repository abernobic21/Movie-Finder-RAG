import pytest
from conftest import WhitespaceTokenizer

from movie_finder.chunking.token import TokenChunker
from movie_finder.models import Document


def doc(plot):
    return Document(movie_id="movie_2000", title="Movie", year=2000, plot=plot)


def chunker(chunk_size, overlap):
    return TokenChunker(WhitespaceTokenizer(), chunk_size=chunk_size, overlap=overlap)


def test_windows_overlap():
    chunks = chunker(4, 2).chunk(doc("a b c d e f g h"))
    assert [c.text for c in chunks] == ["a b c d", "c d e f", "e f g h"]


def test_short_plot_gives_one_chunk():
    chunks = chunker(10, 2).chunk(doc("a b c"))
    assert [c.text for c in chunks] == ["a b c"]


def test_no_trailing_chunk_contained_in_previous_one():
    chunks = chunker(4, 2).chunk(doc("a b c d"))
    assert len(chunks) == 1


def test_every_word_is_covered():
    words = [f"w{i}" for i in range(101)]
    chunks = chunker(7, 3).chunk(doc(" ".join(words)))
    assert {word for c in chunks for word in c.text.split()} == set(words)


def test_original_text_is_preserved():
    chunks = chunker(10, 0).chunk(doc("First paragraph.\n\nSecond paragraph."))
    assert chunks[0].text == "First paragraph.\n\nSecond paragraph."


def test_chunks_carry_movie_details():
    first, second = chunker(2, 0).chunk(doc("a b c d"))
    assert (first.chunk_id, first.position, first.movie_id, first.title, first.year) == (
        "movie_2000#0", 0, "movie_2000", "Movie", 2000,
    )
    assert (second.chunk_id, second.position) == ("movie_2000#1", 1)


def test_name_reflects_settings():
    assert chunker(200, 40).name == "token200_o40"


@pytest.mark.parametrize("chunk_size, overlap", [(0, 0), (10, 10), (10, -1)])
def test_invalid_settings_are_rejected(chunk_size, overlap):
    with pytest.raises(ValueError):
        chunker(chunk_size, overlap)
