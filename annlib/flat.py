import time
import numpy as np
from .base import Index
from .metrics import l2_distances, top_k


class FlatIndex(Index):
    """Exhaustive scan. Always recall 1.0."""

    name = "flat"

    def build(self, vectors):
        t0 = time.perf_counter()
        self.data = vectors
        self.build_seconds = time.perf_counter() - t0

    def search(self, q, k=10):
        d = l2_distances(self.data, q)
        return top_k(d, k)

    def params_str(self):
        return "exact"