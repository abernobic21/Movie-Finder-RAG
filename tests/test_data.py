import csv

import pytest

from movie_finder.data import (
    clean_plot,
    clean_title,
    load_raw,
    make_movie_id,
    read_jsonl,
    sample_movies,
    write_jsonl,
)
from movie_finder.models import Document

LONG_PLOT = "word " * 60


def write_csv(path, rows):
    columns = ["Release Year", "Title", "Origin/Ethnicity", "Director", "Cast", "Genre", "Wiki Page", "Plot"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(rows)


def row(title="Movie", year="1999", plot=LONG_PLOT, genre="drama"):
    return {"Release Year": year, "Title": title, "Origin/Ethnicity": "American", "Genre": genre, "Plot": plot}


def doc(movie_id):
    return Document(movie_id=movie_id, title=movie_id, year=2000, plot="plot")


def test_movie_id_separates_remakes():
    assert make_movie_id("King Kong", 1933) == "king_kong_1933"
    assert make_movie_id("King Kong", 2005) == "king_kong_2005"


def test_movie_id_handles_accents_and_symbols():
    assert make_movie_id("Amélie: A Story!", 2001) == "amelie_a_story_2001"
    assert make_movie_id("$", 1971) == "untitled_1971"


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Aviator, The", "The Aviator"),
        ("Chorus Line, A", "A Chorus Line"),
        ("Battery, TheThe Battery", "The Battery"),
        ("Room (2015 film)", "Room"),
        ("Magic Island (film)", "Magic Island"),
        ("Dena Paona (film)'[1]", "Dena Paona"),
        ("  Jaws  ", "Jaws"),
        ("Crouching Tiger, Hidden Dragon", "Crouching Tiger, Hidden Dragon"),
        ("Film (film)", "Film"),
    ],
)
def test_clean_title(raw, expected):
    assert clean_title(raw) == expected


def test_clean_plot_removes_citations_and_keeps_paragraphs():
    assert clean_plot("First  part.[1]\r\n\r\n\r\nSecond   part.[23]") == "First part.\n\nSecond part."


def test_load_raw_drops_bad_rows(tmp_path):
    path = tmp_path / "raw.csv"
    write_csv(path, [row(), row(title="  "), row(title="B", year="n/a"), row(title="C", plot="too short")])

    movies, dropped = load_raw(path)

    assert [m.movie_id for m in movies] == ["movie_1999"]
    assert dropped == {"bad_year": 1, "empty_title": 1, "short_plot": 1, "duplicate_id": 0}


def test_load_raw_keeps_longer_plot_for_duplicate_id(tmp_path):
    path = tmp_path / "raw.csv"
    longer = LONG_PLOT + "extra"
    write_csv(path, [row(), row(plot=longer), row(plot=LONG_PLOT)])

    movies, dropped = load_raw(path)

    assert len(movies) == 1
    assert movies[0].plot.endswith("extra")
    assert dropped["duplicate_id"] == 2


def test_load_raw_treats_unknown_genre_as_missing(tmp_path):
    path = tmp_path / "raw.csv"
    write_csv(path, [row(genre="unknown")])

    movies, _ = load_raw(path)

    assert movies[0].genre is None


def test_load_raw_reports_missing_columns(tmp_path):
    path = tmp_path / "raw.csv"
    path.write_text("Title,Plot\nA,B\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing columns"):
        load_raw(path)


def test_sample_is_repeatable_regardless_of_input_order():
    movies = [doc(f"movie_{i}") for i in range(100)]

    first = sample_movies(movies, 10, seed=42)
    second = sample_movies(list(reversed(movies)), 10, seed=42)

    assert first == second
    assert first != sample_movies(movies, 10, seed=7)


def test_sample_larger_than_input_returns_everything():
    assert len(sample_movies([doc("a"), doc("b")], 10, seed=42)) == 2


def test_jsonl_round_trip(tmp_path):
    path = tmp_path / "nested" / "movies.jsonl"
    movies = [Document(movie_id="amelie_2001", title="Amélie", year=2001, plot="a\n\nb", genre=None, origin="French")]

    write_jsonl(movies, path)

    assert read_jsonl(path) == movies
