import numpy as np

def l2_distances(db: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Squared L2 from every row of db to q
    SIFT is a euclidian dataset so don't normalize the vectors
    and don't use the cosine here"""
    diff = db - q                                                   # broadcasts q cross rows
    return np.einsum("ij,ij->i", diff, diff)    # faster than (diff**2).sum(1)

def top_k(distances: np.ndarray, k: int) -> np.ndarray:
    """Indices  of the k smalest distances, sorted ascending

    argpartition is O(n) and only guarantees the k smallest land in
    the first k slots unordered. We then sort just those k, which is
    much cheaper than sorting all n"""
    k = min(k, len(distances))
    part = np.argpartition(distances, k - 1)[:k]
    return part[np.argsort(distances[part])]

def recall_at_k(returned: np.ndarray, truth: np.ndarray, k: int = 10) -> float:
    """Fraction of the true top-k that the index actually returned.

    Set intersection, so ordering within the top k does not matter.
    That is the standard ann-benchmarks definition."""
    got = set(np.asarray(returned)[:k].tolist())
    want = set(np.asarray(truth)[:k].tolist())
    return len(got & want) / k