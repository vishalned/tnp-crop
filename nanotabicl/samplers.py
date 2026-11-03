"""How many points a training table gets: the distribution of `n_points` between a lower end and the cap.

A sampler is called as `sampler(rng, lo, hi)` by `CropTables.sample_batch` with the lower end `lo` (the sampler's own `min_points` when set, else
the table config's `min_points`) and the cap `hi` (the smallest of the region's size, `max_points` and what fits in `max_rows`). Chosen in the
Hydra config group `sampler`.
"""

from typing import Optional

import numpy as np


class UniformPoints:
    """Uniform over [lo, hi]: mostly large tables (the default)."""

    def __init__(self, min_points: Optional[int] = None):
        self.min_points = min_points

    def __call__(self, rng: np.random.Generator, lo: int, hi: int) -> int:
        return int(rng.integers(lo, hi + 1))


class LogUniformPoints:
    """Log-uniform over [lo, hi]: each factor of 2 is equally likely, so many small tables and few large ones."""

    def __init__(self, min_points: Optional[int] = None):
        self.min_points = min_points

    def __call__(self, rng: np.random.Generator, lo: int, hi: int) -> int:
        return min(max(int(np.exp(rng.uniform(np.log(lo), np.log(hi + 1)))), lo), hi)
