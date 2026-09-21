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
from annlib.ivf import IVFIndex


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


def test_ivf_full_probe_is_exact(small):
    """nprobe == nlist scans every cluster, so IVF must match flat."""
    train, test, truth = small
    ix = IVFIndex(nlist=64)
    ix.build(train)
    ix.set_params(nprobe=64)
    r = np.mean([recall_at_k(ix.search(q), t) for q, t in zip(test, truth)])
    assert r > 0.999


def test_ivf_recall_is_monotonic(small):
    """More probes must never mean worse recall. If this fails, the
    inverted lists or the centroid ranking are wrong."""
    train, test, truth = small
    ix = IVFIndex(nlist=64)
    ix.build(train)
    prev = -1.0
    for nprobe in (1, 4, 16, 64):
        ix.set_params(nprobe=nprobe)
        r = np.mean([recall_at_k(ix.search(q), t) for q, t in zip(test, truth)])
        assert r >= prev - 1e-9
        prev = r