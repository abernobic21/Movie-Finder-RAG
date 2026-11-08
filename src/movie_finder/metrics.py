def rank_of(movie_id: str, ranked_ids: list[str]) -> int | None:
    """1-based position of movie_id in the results, or None if it is missing."""
    try:
        return ranked_ids.index(movie_id) + 1
    except ValueError:
        return None


def recall_at_k(ranks: list[int | None], k: int) -> float:
    if not ranks:
        raise ValueError("No queries to evaluate")
    return sum(1 for rank in ranks if rank is not None and rank <= k) / len(ranks)


def mrr(ranks: list[int | None]) -> float:
    if not ranks:
        raise ValueError("No queries to evaluate")
    return sum(1 / rank for rank in ranks if rank is not None) / len(ranks)
