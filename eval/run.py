import argparse
import json
import time
from pathlib import Path

from movie_finder.config import ComponentConfig, load_config
from movie_finder.metrics import mrr, rank_of, recall_at_k
from movie_finder.pipeline import build_pipeline, get_client

TOP_K = 10


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure retrieval quality on the eval queries")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--retriever", choices=["vector", "bm25", "hybrid"], help="Override the config's retriever")
    parser.add_argument("--split", choices=["tune", "test"], default="tune")
    args = parser.parse_args()

    config = load_config(args.config)
    name = config.name
    if args.retriever:
        config.retriever = ComponentConfig(type=args.retriever)
        name = f"{name}_{args.retriever}"

    queries = [
        json.loads(line)
        for line in Path("eval/queries.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    queries = [q for q in queries if q["split"] == args.split and q["movie_id"] is not None]

    pipeline = build_pipeline(config, get_client())
    details = []
    start = time.perf_counter()
    for q in queries:
        results = pipeline.search(q["query"], top_k=TOP_K)
        rank = rank_of(q["movie_id"], [movie.movie_id for movie in results])
        details.append({**q, "rank": rank, "top": [movie.movie_id for movie in results[:5]]})
    ms_per_query = (time.perf_counter() - start) * 1000 / len(queries)

    ranks = [d["rank"] for d in details]
    summary = {
        "name": name,
        "split": args.split,
        "queries": len(queries),
        "recall@1": recall_at_k(ranks, 1),
        "recall@5": recall_at_k(ranks, 5),
        "recall@10": recall_at_k(ranks, 10),
        f"mrr@{TOP_K}": mrr(ranks),
        "ms_per_query": round(ms_per_query, 1),
    }

    Path("results/raw").mkdir(parents=True, exist_ok=True)
    Path(f"results/{name}_{args.split}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    Path(f"results/raw/{name}_{args.split}.json").write_text(json.dumps(details, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
