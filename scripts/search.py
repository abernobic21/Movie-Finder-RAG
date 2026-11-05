import argparse
from pathlib import Path

from movie_finder.config import ComponentConfig, load_config
from movie_finder.pipeline import build_pipeline, get_client


def main() -> None:
    parser = argparse.ArgumentParser(description="Try a search from the command line")
    parser.add_argument("query")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--retriever", choices=["vector", "bm25", "hybrid"], help="Override the config's retriever")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.retriever:
        config.retriever = ComponentConfig(type=args.retriever)

    for rank, movie in enumerate(build_pipeline(config, get_client()).search(args.query, args.top_k), start=1):
        print(f"{rank}. {movie.title} ({movie.year})  score={movie.score:.4f}")
        print(f"   {movie.passages[0].chunk.text[:200]!r}")


if __name__ == "__main__":
    main()
