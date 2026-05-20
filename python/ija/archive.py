"""
Elite archive for the Immortal Jellyfish Algorithm.

Maintains the best solutions found during the search.  When the archive
exceeds its capacity, entries with the smallest crowding distance (i.e.
those clustered near neighbours in objective space) are pruned first,
which helps preserve diversity among the retained elites.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .operators import crowding_distance


class EliteArchive:
    """Fixed-capacity archive of elite solutions with diversity-aware pruning."""

    def __init__(self, max_size: int = 15) -> None:
        self.max_size = max_size
        self._positions: List[np.ndarray] = []
        self._fitnesses: List[float] = []

    def update(self, population: np.ndarray, fitnesses: np.ndarray) -> None:
        """Merge new candidates into the archive and prune to max_size."""
        for pos, fit in zip(population, fitnesses):
            self._positions.append(pos.copy())
            self._fitnesses.append(float(fit))

        if len(self._positions) <= self.max_size:
            return

        fits = np.array(self._fitnesses)
        cd = crowding_distance(fits)

        combined = sorted(
            zip(fits, cd, self._positions),
            key=lambda x: (x[0], -x[1]),
        )
        combined = combined[: self.max_size]
        self._fitnesses = [c[0] for c in combined]
        self._positions = [c[2].copy() for c in combined]

    def sample(self) -> Optional[np.ndarray]:
        """Return a uniformly random archive member, or None if empty."""
        if not self._positions:
            return None
        idx = np.random.randint(0, len(self._positions))
        return self._positions[idx]

    def best(self) -> Optional[np.ndarray]:
        """Return the archive member with the lowest fitness."""
        if not self._positions:
            return None
        return self._positions[int(np.argmin(self._fitnesses))]

    def best_fitness(self) -> float:
        """Return the lowest fitness value in the archive."""
        if not self._fitnesses:
            return np.inf
        return min(self._fitnesses)

    def __len__(self) -> int:
        return len(self._positions)

    def is_populated(self) -> bool:
        return len(self._positions) > 0
