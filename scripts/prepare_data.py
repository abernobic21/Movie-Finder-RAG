import argparse
import sys
from pathlib import Path

from movie_finder.data import load_raw, sample_movies, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean the raw dataset and create a fixed subset")
    parser.add_argument("--input", type=Path, default=Path("data/raw/wiki_movie_plots_deduped.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--subset-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.input.exists():
        sys.exit(f"{args.input} not found. Download the dataset first, see README.")

    movies, dropped = load_raw(args.input)
    subset = sample_movies(movies, args.subset_size, args.seed)

    write_jsonl(movies, args.output_dir / "movies.jsonl")
    write_jsonl(subset, args.output_dir / f"movies_{args.subset_size}.jsonl")

    years = [m.year for m in movies]
    print(f"Clean movies: {len(movies)} ({min(years)}-{max(years)})")
    print(f"Subset: {len(subset)} (seed {args.seed})")
    print(f"Dropped or replaced: {dropped}")


if __name__ == "__main__":
    main()
