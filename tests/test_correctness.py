"""Correctness gates. Run these after touching any index.

Each index has one setting where it must degenerate to exact search.
That property catches almost every real bug, especially the id
mapping mistakes that otherwise look like mysterious low recall.
"""
import numpy as np
import pytest
from annlib.dataset import load_sift
from annlib.metrics import recall_at_k
from annlib.flat import FlatIndex


@pytest.fixture(scope="module")
def small():
    # 50k vectors, 100 queries
    return load_sift(n=50_000, n_queries=100)


def test_flat_is_exact(small):
    train, test, truth = small
    ix = FlatIndex()
    ix.build(train)
    r = np.mean([recall_at_k(ix.search(q), t) for q, t in zip(test, truth)])
    assert r == 1.0
