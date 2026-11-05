# Movie Finder

A RAG system that identifies movies from vague descriptions, with measured comparisons of chunking, embedding and retrieval methods.

Work in progress.

## Setup

```bash
conda create -n movie-finder python=3.11
conda activate movie-finder
pip install -e ".[dev]"
```

## Data

The project uses [Wikipedia Movie Plots](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots) from Kaggle. It isn't included in this repository.

1. Download the dataset from Kaggle.
2. Put `wiki_movie_plots_deduped.csv` in `data/raw/`.
3. Prepare it:

```bash
python scripts/prepare_data.py
```

This writes all cleaned movies to `data/processed/movies.jsonl` and a fixed random subset to `data/processed/movies_5000.jsonl`.

## Search

Start Qdrant, build the index for a config, then search:

```bash
docker compose up -d qdrant
python scripts/build_index.py configs/baseline.yaml
python scripts/search.py "a boy hides a stranded visitor from space"
python scripts/search.py "a boy hides a stranded visitor from space" --retriever bm25
```

An existing index is reused when the chunking and embedding settings match. If the data itself changes, add `--rebuild`.

## Tests

```bash
pytest
```
