import json
import random
from collections import Counter
from pathlib import Path

from movie_finder.data import read_jsonl

TUNE_FRACTION = 0.3
SEED = 2026
TYPES = {"generated", "memory", "absent"}


def read_queries(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


movie_ids = {movie.movie_id for movie in read_jsonl(Path("data/processed/movies_5000.jsonl"))}
queries = [{**q, "type": "generated"} for q in read_queries(Path("eval/queries_draft.jsonl"))]
queries += read_queries(Path("eval/queries_manual.jsonl"))

for q in queries:
    if q["type"] not in TYPES:
        raise ValueError(f"Unknown type {q['type']!r} in query: {q['query']}")
    if q["type"] == "absent":
        if q["movie_id"] is not None:
            raise ValueError(f"Absent query must have movie_id null: {q['query']}")
    elif q["movie_id"] not in movie_ids:
        raise ValueError(f"Unknown movie_id {q['movie_id']!r} in query: {q['query']}")

rng = random.Random(SEED)
for query_type in sorted(TYPES):
    group = [q for q in queries if q["type"] == query_type]
    rng.shuffle(group)
    n_tune = round(len(group) * TUNE_FRACTION)
    for i, q in enumerate(group):
        q["split"] = "tune" if i < n_tune else "test"

with Path("eval/queries.jsonl").open("w", encoding="utf-8") as f:
    for q in queries:
        record = {"query": q["query"], "movie_id": q["movie_id"], "type": q["type"], "split": q["split"]}
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

for (query_type, split), count in sorted(Counter((q["type"], q["split"]) for q in queries).items()):
    print(f"{query_type:<10} {split:<5} {count}")
print(f"total      {len(queries)}")