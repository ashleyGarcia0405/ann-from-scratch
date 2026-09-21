import time
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from .base import Index
from .metrics import l2_distances, top_k

class IVFIndex(Index):
    """Inverted file index.

    Partition the database into nlist clusters with k-means. At query
    time rank the centroids, then scan only the vectors inside the
    nprobe closest clusters. nprobe is the recall/speed knob"""

    name = "ivf"

    def __init__(self, nlist=1024, train_size=100_000, seed=42, contiguous=True):
        self.nlist = nlist
        # k-means on a sample. Fitting on all 1M vectors takes far longer
        # and moves the centroids very little
        self.train_size = train_size
        self.seed = seed
        # Set False to build the slow variant that gathers scattered rows
        # instead of reordering. Used for the memory-layout exp in writeup
        self.contiguous = contiguous
        self.nprobe = 1

    def build(self, vectors):
        t0 = time.perf_counter()
        rng = np.random.default_rng(self.seed)

        # fit the centroids on a random sample
        sample_n = min(self.train_size, len(vectors))
        sample = vectors[rng.choice(len(vectors), sample_n, replace=False)]

        km = MiniBatchKMeans(
            n_clusters=self.nlist,
            batch_size=10_000,
            n_init=3,
            random_state=self.seed,
        ).fit(sample)
        self.centroids = km.cluster_centers_.astype(np.float32)

        # assign every vector to its nearest centroid, in chunks
        assign = np.empty(len(vectors), dtype=np.int32)
        cent_sq = (self.centroids ** 2).sum(axis=1)
        for s in range(0, len(vectors), 10_000):
            chunk = vectors[s:s + 10_000]
            # squared distance expanded so we never materialise
            # (chunk, nlist, dim). The constant chunk term is dropped
            # because it does not affect the argmin.
            d = cent_sq[None, :] - 2.0 * (chunk @ self.centroids.T)
            assign[s:s + 10_000] = np.argmin(d, axis=1)

        # physically reorder the data so each cluster is one contiguous block
        # this makes the scan fast bc a slice is a sequential memory read while
        # fancy indexing jumps around and thrashes cache
        if self.contiguous:
            order = np.argsort(assign, kind="stable")
            self.data = np.ascontiguousarray(vectors[order])
            self.ids = order.astype(np.int32)  # row -> original id
            sorted_assign = assign[order]
            # offsets[c] .. offsets[c+1] is cluster c's block
            counts = np.bincount(sorted_assign, minlength=self.nlist)
            self.offsets = np.zeros(self.nlist + 1, dtype=np.int64)
            np.cumsum(counts, out=self.offsets[1:])
        else:
            # Variant for the experiment: keep original order, store
            # a list of id arrays per cluster.
            self.data = vectors
            self.lists = [np.where(assign == c)[0].astype(np.int32)
                          for c in range(self.nlist)]

        self.build_seconds = time.perf_counter() - t0

    def search(self, q, k=10):
        # Rank centroids. This costs nlist distance computations,
        # which is why very large nlist starts to hurt latency even
        # at low nprobe.
        cd = l2_distances(self.centroids, q)
        probe = top_k(cd, self.nprobe)

        if self.contiguous:
            parts, id_parts = [], []
            for c in probe:
                s, e = self.offsets[c], self.offsets[c + 1]
                parts.append(self.data[s:e])  # contiguous slice
                id_parts.append(self.ids[s:e])
            cand = np.concatenate(parts)
            cand_ids = np.concatenate(id_parts)
        else:
            cand_ids = np.concatenate([self.lists[c] for c in probe])
            cand = self.data[cand_ids]  # scattered gather

        d = l2_distances(cand, q)
        local = top_k(d, k)
        # Map back to original dataset ids. Returning `local` here is
        # the single most common bug in this project and produces
        # near-zero recall that looks like an algorithm problem.
        return cand_ids[local]

    def params_str(self):
        return f"nlist={self.nlist},nprobe={self.nprobe}"