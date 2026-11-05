import csv
import json
import random
import re
import unicodedata
from dataclasses import asdict
from pathlib import Path

from movie_finder.models import Document

REQUIRED_COLUMNS = {"Release Year", "Title", "Origin/Ethnicity", "Genre", "Plot"}
MIN_PLOT_WORDS = 50


def make_movie_id(title: str, year: int) -> str:
    """Title + year, so that remakes get different IDs."""
    ascii_title = unicodedata.normalize("NFKD", title)
    ascii_title = "".join(c for c in ascii_title if not unicodedata.combining(c))
    slug = re.sub(r"\W+", "_", ascii_title.casefold()).strip("_")
    return f"{slug or 'untitled'}_{year}"


def clean_title(title: str) -> str:
    """Fixes 'Aviator, The', 'Battery, TheThe Battery' and 'Room (2015 film)'."""
    title = re.sub(r"\[\d+\]", "", title)
    title = re.sub(r"\s*\((\d{4}\s+)?film\)'?", "", title, flags=re.IGNORECASE).strip()
    moved_article = re.fullmatch(r"(.+), (The|A|An)(?:\2 \1)?", title)
    return f"{moved_article[2]} {moved_article[1]}" if moved_article else title


def clean_plot(plot: str) -> str:
    """Remove citation markers and extra whitespace, keeping paragraph breaks."""
    plot = re.sub(r"\[\d+\]", "", plot)
    lines = (re.sub(r"[ \t]+", " ", line).strip() for line in plot.splitlines())
    return "\n\n".join(line for line in lines if line)


def _optional(value: str) -> str | None:
    value = value.strip()
    return None if not value or value.casefold() == "unknown" else value


def load_raw(path: Path) -> tuple[list[Document], dict[str, int]]:
    """Load and clean the Kaggle CSV. Returns the movies and how many rows were dropped and why."""
    csv.field_size_limit(10_000_000)
    movies: dict[str, Document] = {}
    dropped = {"bad_year": 0, "empty_title": 0, "short_plot": 0, "duplicate_id": 0}

    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")

        for row in reader:
            title = clean_title(row["Title"])
            plot = clean_plot(row["Plot"])
            if not title:
                dropped["empty_title"] += 1
                continue
            try:
                year = int(row["Release Year"])
            except ValueError:
                dropped["bad_year"] += 1
                continue
            if len(plot.split()) < MIN_PLOT_WORDS:
                dropped["short_plot"] += 1
                continue

            doc = Document(
                movie_id=make_movie_id(title, year),
                title=title,
                year=year,
                plot=plot,
                genre=_optional(row["Genre"]),
                origin=_optional(row["Origin/Ethnicity"]),
            )
            # Same title and year: keep the longer plot, so every ID stays unique
            existing = movies.get(doc.movie_id)
            if existing is not None:
                dropped["duplicate_id"] += 1
                if len(existing.plot) >= len(doc.plot):
                    continue
            movies[doc.movie_id] = doc

    return list(movies.values()), dropped


def sample_movies(movies: list[Document], size: int, seed: int) -> list[Document]:
    """Same seed and input give the same subset, regardless of input order."""
    ordered = sorted(movies, key=lambda m: m.movie_id)
    return random.Random(seed).sample(ordered, min(size, len(ordered)))


def write_jsonl(movies: list[Document], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for movie in movies:
            f.write(json.dumps(asdict(movie), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[Document]:
    with path.open(encoding="utf-8") as f:
        return [Document(**json.loads(line)) for line in f]
