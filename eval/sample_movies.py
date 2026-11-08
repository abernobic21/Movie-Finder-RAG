from pathlib import Path

from movie_finder.data import read_jsonl, sample_movies, write_jsonl

BATCH_SIZE = 20

movies = read_jsonl(Path("data/processed/movies_5000.jsonl"))
sample = sample_movies(movies, size=80, seed=2026)

out = Path("data/eval")
out.mkdir(parents=True, exist_ok=True)
for i in range(0, len(sample), BATCH_SIZE):
    write_jsonl(sample[i : i + BATCH_SIZE], out / f"batch_{i // BATCH_SIZE + 1}.jsonl")
