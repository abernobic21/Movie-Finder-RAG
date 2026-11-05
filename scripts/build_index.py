import argparse
from pathlib import Path

from movie_finder.config import load_config
from movie_finder.pipeline import build_index, get_client


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk and embed the data for one config")
    parser.add_argument("config", type=Path)
    parser.add_argument("--rebuild", action="store_true", help="Rebuild even if the index already exists")
    args = parser.parse_args()

    collection, built = build_index(load_config(args.config), get_client(), rebuild=args.rebuild)
    print(f"{'Built' if built else 'Already built, reusing'}: {collection}")


if __name__ == "__main__":
    main()
