import os
import h5py
import numpy as np

# train shape: (1000000, 128) - db
# test shape: 10000, 128) - queries
# neighbors: (10000, 100) - the true 1-- nearest neighbors per query
# distances: matching distances
# Anchored to the repo root so paths hold whatever the working directory is
# (pytest runners often cd into tests/).
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
DEFAULT_PATH = os.path.join(DATA_DIR, "sift-128-euclidean.hdf5")
CACHE_DIR = os.path.join(DATA_DIR, "cache")

def load_raw(path=DEFAULT_PATH):
    with h5py.File(path, "r") as f:
        train = np.asarray(f["train"], dtype=np.float32)
        test = np.asarray(f["test"], dtype=np.float32)
        truth = np.asarray(f["neighbors"], dtype=np.int32)
    return train, test, truth

def exact_neighbors(train, queries, k=100, chunk=256):
    """ Brute-force ground truth. Used when the provided truth
    does not apply because train has to be subsampled """
    out = np.empty((len(queries), k), dtype=np.int32)
    train_sq = (train ** 2).sum(axis=1)
    for i in range(0, len(queries), chunk):
        q = queries[i:i + chunk]
        # the squared Euclidean distance from every query in the chunk to every vector in db
        # || a - b ||^2 = (a - b)(a - b) = ||a||^2 - 2(ab) + ||b||^2
        d = train_sq[None, :] - 2.0 * (q @ train.T) + ( q ** 2).sum(axis=1)[:, None]
        idx = np.argpartition(d, k, axis=1)[:, :k]
        rows = np.arange(len(q))[:, None]
        order = np.argsort(d[rows, idx], axis=1)
        out[i:i + chunk] = idx[rows, order]
    return out

def load_sift(path=DEFAULT_PATH, n=None, n_queries=None, k=100):
    """Return train, test, truth with truth always valid for the
        returned train. Subset ground truth is cached on first use."""
    train, test, truth = load_raw(path)

    if n_queries is not None:
        test = test[:n_queries]
        truth = truth[:n_queries]

    if n is None or n >= len(train):
        return train, test, truth

    train = train[:n]
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"truth_n{n}_q{len(test)}_k{k}.npy")
    if os.path.exists(cache):
        truth = np.load(cache)
    else:
        truth = exact_neighbors(train, test, k=k)
        np.save(cache, truth)
    return train, test, truth