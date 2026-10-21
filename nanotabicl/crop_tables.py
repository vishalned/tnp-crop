"""Tables of WOFOST-simulated crop seasons for NanoTabICL.

Input: the processed table from `src/data_pipeline/wofost/process_wofost_dataset.py`
(`data/processed/wofost_{crop}_daily.parquet`, `--align calendar`): one row
per location x sowing year x jitter, with static features, daily
weather/crop columns (`{feature}_d{day:03d}`) over CY-Bench's window and the
yield.

Each row becomes one feature vector: CY-Bench's tabular features
(`cybench_features.py`: 30-day blocks counted back from the horizon's
cutoff, the benchmark's statistics, column names and order), standardised
with the benchmark's normaliser fitted on the train years, columns with
more than 5% NaN dropped, remaining NaN set to 0 (= the variable's mean).
The benchmark side (`NanoTabICLRegressor`) fills NaN the same way.

Regions (countries) with fewer than `min_points` points are dropped. A
training table always comes from one region and uses one jitter run per
(point, year) cell, like a real table would. Two ways to split it into
context and query rows (`episodes`):

- "structured" (forecasting): `n_points` of the region's points (from
  `min_points` up to all of them), a target year T and per point
  `n_context_years` distinct earlier years. Context = those cells with their
  yields, queries = the points at year T. This is the evaluation task.
- "random" (how TabICL itself is pretrained and fine-tuned): `n_points` of
  the region's points at every train year, and a random subset of the rows
  as context, the rest as queries -- no temporal structure.

Within a batch all tables share their shape (number of rows and context
rows), so batches need no padding. The batch's size is drawn from the range
of a region picked uniformly at random; the other tables of the batch come
from regions large enough for that size.

Worlds datasets (`generate_worlds_dataset`, several sampled worlds per location): a point is a world
(`location_index`), and `site_index` is its location. A training table then has at most one world per
site, so a table never holds two simulations of the same place; region sizes (`min_points`, the number
of points per table) count sites. Tables without `site_index` are sampled exactly as before.

Year split by sowing year: training tables only use the train years;
evaluation is walk-forward (T = a val/test year, context = every earlier
year of the region's points).
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch

from nanotabicl.cybench_features import MAX_COLUMN_NAN_RATE, apply_normalizer, cybench_features, fit_normalizer


@dataclass
class CropTableConfig:
    train_years: Sequence[int] = (2005, 2016)
    countries: Optional[Sequence[str]] = None  # None = every country in the table
    target: str = "yield_t_per_ha"
    horizon: str = "eos"  # CY-Bench's end_of_sequence: eos | middle-of-season | quarter-of-season | early-season | eos-N
    episodes: str = "structured"  # "structured" | "random"
    min_points: int = 10  # also: regions with fewer points are dropped
    max_points: Optional[int] = None  # None = up to all of a region's points
    max_rows: int = 4096  # cap on rows per table (memory)
    min_context_years: int = 5  # structured episodes
    max_context_years: int = 11
    min_train_fraction: float = 0.5  # random episodes: share of rows used as context
    max_train_fraction: float = 0.9


class CropTables:
    def __init__(self, path: str, config: CropTableConfig = CropTableConfig()):
        if config.episodes not in ("structured", "random"):
            raise ValueError(f"episodes must be 'structured' or 'random', got {config.episodes!r}.")
        self.config = config
        df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
        if "reached_maturity" in df:
            df = df[df["reached_maturity"].astype(bool)]
        df = df[np.isfinite(df[config.target])].reset_index(drop=True)
        if "country" not in df:
            df["country"] = "all"
        self.df = df

        self.features, self.feature_names = self._features(df)
        self.y = df[config.target].to_numpy(np.float32)
        self.point = df["location_index"].to_numpy()
        self.year = df["sowing_year"].to_numpy()

        self.cells = {key: idx.to_numpy() for key, idx in df.groupby(["location_index", "sowing_year"]).groups.items()}
        self.point_years = {}
        for p, y in self.cells:
            self.point_years.setdefault(p, set()).add(y)
        countries = df.groupby("location_index")["country"].first()
        # site of each point: its location (worlds datasets), else the point itself
        self.site_of = df.groupby("location_index")["site_index"].first().to_dict() if "site_index" in df else None
        wanted = config.countries or sorted(countries.unique())
        regions = {c: np.array(countries.index[countries == c]) for c in wanted}
        n_sites = {c: self._n_sites(p) for c, p in regions.items()}
        self.dropped_regions = {c: n_sites[c] for c in regions if n_sites[c] < config.min_points}
        self.country_points = {c: p for c, p in regions.items() if n_sites[c] >= config.min_points}
        self.country_sites = {c: n_sites[c] for c in self.country_points}
        if not self.country_points:
            raise ValueError(f"No region has >= {config.min_points} points ({ {c: len(p) for c, p in regions.items()} }).")
        self.years = sorted(df["sowing_year"].unique())
        self.train_pool = [y for y in self.years if config.train_years[0] <= y <= config.train_years[1]]

    def _features(self, df: pd.DataFrame) -> tuple:
        cfg = self.config
        raw, names = cybench_features(df, horizon=cfg.horizon, drop_sparse=False)
        # as the benchmark: normalise (stats pooled over all blocks), then drop sparse columns
        train = df["sowing_year"].between(cfg.train_years[0], cfg.train_years[1]).to_numpy()
        self.normalizer = fit_normalizer(raw, names, rows=train)
        features = apply_normalizer(raw, names, self.normalizer)
        keep = np.isnan(raw).mean(axis=0) <= MAX_COLUMN_NAN_RATE
        names = [n for n, k in zip(names, keep) if k]
        return np.nan_to_num(features[:, keep], nan=0.0), names

    # ----- training batches -------------------------------------------------------------

    def sample_batch(self, rng: np.random.Generator, batch_size: int, max_attempts: int = 1000) -> dict:
        """A training batch {"x": [B, rows, features], "y": [B, rows], "n_train"} of same-shape tables."""
        cfg = self.config
        countries = list(self.country_points)
        lead = countries[rng.integers(len(countries))]
        rows_per_point = (int(rng.integers(cfg.min_context_years, cfg.max_context_years + 1)) + 1
                          if cfg.episodes == "structured" else len(self.train_pool))
        hi = min(self.country_sites[lead], cfg.max_points or np.inf, cfg.max_rows // rows_per_point)
        n_points = int(rng.integers(cfg.min_points, max(hi, cfg.min_points) + 1))
        eligible = [c for c in countries if self.country_sites[c] >= n_points]
        n_train = None
        if cfg.episodes == "random":
            n_rows = n_points * rows_per_point
            n_train = int(n_rows * rng.uniform(cfg.min_train_fraction, cfg.max_train_fraction))

        tables = []
        for attempt in range(max_attempts):
            country = lead if attempt == 0 else eligible[rng.integers(len(eligible))]
            if cfg.episodes == "structured":
                table = self._structured(rng, country, n_points, rows_per_point - 1)
            else:
                table = self._random(rng, country, n_points, n_train)
            if table is not None:
                tables.append(table[0])
                n_train = table[1]
                if len(tables) == batch_size:
                    idx = np.stack(tables)
                    return {"x": torch.from_numpy(self.features[idx]), "y": torch.from_numpy(self.y[idx]), "n_train": n_train}
        raise RuntimeError(f"Couldn't build {batch_size} tables of {n_points} points; check min_points / the data.")

    def _structured(self, rng, country, n_points, n_context_years) -> Optional[tuple]:
        targets = [T for T in self.train_pool if sum(y < T for y in self.train_pool) >= n_context_years]
        if not targets:
            raise ValueError(f"No train year has {n_context_years} earlier train years; lower max_context_years.")
        T = int(rng.choice(targets))
        before = {y for y in self.train_pool if y < T}
        candidates = [p for p in self.country_points[country]
                      if T in self.point_years[p] and len(self.point_years[p] & before) >= n_context_years]
        points = self._choose_points(rng, candidates, n_points)
        if points is None:
            return None
        context, query = [], []
        for p in points:
            for y in rng.choice(sorted(self.point_years[p] & before), size=n_context_years, replace=False):
                context.append(rng.choice(self.cells[(p, y)]))
            query.append(rng.choice(self.cells[(p, T)]))
        return np.array(context + query), len(context)

    def _random(self, rng, country, n_points, n_train) -> Optional[tuple]:
        pool = set(self.train_pool)
        candidates = [p for p in self.country_points[country] if pool <= self.point_years[p]]
        points = self._choose_points(rng, candidates, n_points)
        if points is None:
            return None
        rows = np.array([rng.choice(self.cells[(p, y)]) for p in points for y in sorted(pool)])
        return rng.permutation(rows), n_train  # first n_train rows = context

    def _n_sites(self, points) -> int:
        return len(points) if self.site_of is None else len({self.site_of[p] for p in points})

    def _choose_points(self, rng, candidates, n_points) -> Optional[np.ndarray]:
        """`n_points` distinct candidates at random, at most one per site; None if there are too few."""
        if len(candidates) < n_points:
            return None
        if self.site_of is None:
            return rng.choice(candidates, size=n_points, replace=False)
        chosen, sites = [], set()
        for p in rng.permutation(candidates):
            if self.site_of[p] not in sites:
                chosen.append(p)
                sites.add(self.site_of[p])
                if len(chosen) == n_points:
                    return np.array(chosen)
        return None

    # ----- evaluation -------------------------------------------------------------------

    def walk_forward_tables(self, eval_years: Sequence[int], max_context_rows: int = 4000, seed: int = 0):
        """Evaluation tables: for every region and eval year T, context = the region's points at every
        earlier year (one jitter per cell), queries = the same points at T. Points are split into chunks so
        a table has at most `max_context_rows` context rows. Deterministic for a given seed."""
        rng = np.random.default_rng(seed)
        for country, points in self.country_points.items():
            for T in eval_years:
                at_T = [p for p in points if T in self.point_years[p]]
                if not at_T:
                    continue
                n_before = max(len([y for y in self.point_years[p] if y < T]) for p in at_T)
                chunk = max(1, max_context_rows // max(n_before, 1))
                for start in range(0, len(at_T), chunk):
                    context, query = [], []
                    for p in at_T[start : start + chunk]:
                        context += [rng.choice(self.cells[(p, y)]) for y in sorted(self.point_years[p]) if y < T]
                        query.append(rng.choice(self.cells[(p, T)]))
                    if not context:
                        continue
                    idx = np.array(context + query)
                    yield {"x": torch.from_numpy(self.features[idx])[None], "y": torch.from_numpy(self.y[idx])[None],
                           "n_train": len(context), "rows": idx, "country": country, "year": T}
