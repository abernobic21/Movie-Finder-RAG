import pytest

from movie_finder.metrics import mrr, rank_of, recall_at_k

RANKS = [1, 3, None, 2]


def test_rank_of():
    assert rank_of("b", ["a", "b", "c"]) == 2
    assert rank_of("x", ["a", "b", "c"]) is None


def test_recall_at_k_by_hand():
    assert recall_at_k(RANKS, 1) == 0.25
    assert recall_at_k(RANKS, 3) == 0.75


def test_mrr_by_hand():
    assert mrr(RANKS) == pytest.approx((1 + 1 / 3 + 0 + 1 / 2) / 4)


@pytest.mark.parametrize("metric", [lambda r: recall_at_k(r, 5), mrr])
def test_metrics_reject_empty_input(metric):
    with pytest.raises(ValueError):
        metric([])
