"""The contract every index obeys"""

from abc import ABC, abstractmethod
import numpy as np


class Index(ABC):
    name = "base"
    build_seconds = None

    @abstractmethod
    def build(self, vectors: np.ndarray) -> None:
        """Consume the database. Called once"""

    def set_params(self, **kwargs) -> None:
        """Set query-time knobs (nprobe, ef_search). Called many
        times between benchmark runs without rebuilding."""
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def search(self, q: np.ndarray, k: int = 10) -> np.ndarray:
        """Return the ids of the approximate k nearest neighbors,
        closest first. Ids index into the array passed to build()."""

    def params_str(self) -> str:
        """Human-readable current settings. Written to the csv."""
        return ""