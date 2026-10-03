"""Tables of WOFOST-simulated crop seasons for NanoTabICL.

Input: the processed table from `src/data_pipeline/wofost/process_wofost_dataset.py`
(`data/processed/wofost_{crop}_daily.parquet`): one row per location x
sowing year x jitter, with static features, daily weather/crop columns
(`{feature}_d{day:03d}`) and the yield.

Each row becomes one feature vector: the static features plus the daily
series aggregated into fixed buckets (`bucket_days`, e.g. 7 = weekly), with
sums for fluxes (precipitation, radiation, ET0, water balance) and means
for states (temperatures, fpar, soil moisture).

A *table* (one in-context-learning dataset) is built like this:
1. a country, uniformly at random (not weighted by its number of points);
2. `n_points` points of that country;
3. a target year T, and per point `n_context_years` distinct earlier years;
4. one jitter run per (point, year) cell;
rows = the context cells (with yields) followed by the query cells (the
points at year T). Within a batch all tables share `n_points` and
`n_context_years`, so they have the same shape and need no padding.

Year split by sowing year: training tables draw T and context from the
train years only; evaluation is walk-forward (T = a val/test year, context
= every earlier year).
"""

import warnings
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch

STATIC_FEATURES = ["latitude", "longitude", "awc", "bulk_density"]
SUM_FEATURES = {"prec", "rad", "et0", "cwb"}  # fluxes: summed per bucket; everything else: mean


@dataclass
class CropTableConfig:
    train_years: Sequence[int] = (2005, 2016)
    countries: Optional[Sequence[str]] = None  # None = every country in the table
    target: str = "yield_t_per_ha"
    bucket_days: int = 7
    daily_features: Optional[Sequence[str]] = None  # None = every daily feature in the table
    min_points: int = 3
    max_points: int = 10
    min_context_years: int = 5
    max_context_years: int = 11


class CropTables:
    def __init__(self, path: str, config: CropTableConfig = CropTableConfig()):
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
        wanted = config.countries or sorted(countries.unique())
        self.country_points = {c: np.array(countries.index[countries == c]) for c in wanted}
        self.country_points = {c: p for c, p in self.country_points.items() if len(p)}
        self.years = sorted(df["sowing_year"].unique())

    def _features(self, df: pd.DataFrame) -> tuple:
        cfg = self.config
        daily = sorted({c.rsplit("_d", 1)[0] for c in df.columns if "_d" in c and c.rsplit("_d", 1)[1].isdigit()})
        daily = [f for f in daily if cfg.daily_features is None or f in cfg.daily_features]
        blocks, names = [df[STATIC_FEATURES].to_numpy(np.float32)], list(STATIC_FEATURES)
        for f in daily:
            cols = sorted([c for c in df.columns if c.startswith(f"{f}_d") and c[len(f) + 2:].isdigit()])
            values = df[cols].to_numpy(np.float32)
            n = values.shape[1] // cfg.bucket_days
            buckets = values[:, : n * cfg.bucket_days].reshape(len(df), n, cfg.bucket_days)
            with warnings.catch_warnings():  # all-NaN buckets (e.g. soil moisture outside the season)
                warnings.simplefilter("ignore", RuntimeWarning)
                agg = np.nansum(buckets, axis=2) if f in SUM_FEATURES else np.nanmean(buckets, axis=2)
            blocks.append(agg)
            names += [f"{f}_b{k:02d}" for k in range(n)]
        features = np.concatenate(blocks, axis=1)
        return np.nan_to_num(features, nan=0.0), names  # e.g. soil moisture outside the season

    def _table(self, rng, country, T, n_points, n_context_years, pool) -> Optional[tuple]:
        before = [y for y in pool if y < T]
        candidates = [p for p in self.country_points[country]
                      if T in self.point_years[p] and len(self.point_years[p] & set(before)) >= n_context_years]
        if len(candidates) < n_points:
            return None
        points = rng.choice(candidates, size=n_points, replace=False)
        context, query = [], []
        for p in points:
            years = sorted(self.point_years[p] & set(before))
            for y in rng.choice(years, size=n_context_years, replace=False):
                context.append(rng.choice(self.cells[(p, y)]))
            query.append(rng.choice(self.cells[(p, T)]))
        return np.array(context + query), len(context)

    def sample_batch(self, rng: np.random.Generator, batch_size: int, max_attempts: int = 1000) -> dict:
        """A training batch: `batch_size` tables of the same shape, T and context from the train years."""
        cfg = self.config
        pool = [y for y in self.years if cfg.train_years[0] <= y <= cfg.train_years[1]]
        n_points = int(rng.integers(cfg.min_points, cfg.max_points + 1))
        n_context_years = int(rng.integers(cfg.min_context_years, cfg.max_context_years + 1))
        targets = [T for T in pool if sum(y < T for y in pool) >= n_context_years]
        if not targets:
            raise ValueError(f"No train year has {n_context_years} earlier train years; lower max_context_years.")
        countries = list(self.country_points)
        rows = []
        for _ in range(max_attempts):
            table = self._table(rng, countries[rng.integers(len(countries))], int(rng.choice(targets)),
                                n_points, n_context_years, pool)
            if table is not None:
                rows.append(table[0])
                if len(rows) == batch_size:
                    idx = np.stack(rows)
                    return {"x": torch.from_numpy(self.features[idx]), "y": torch.from_numpy(self.y[idx]),
                            "n_train": table[1]}
        raise RuntimeError(f"Couldn't build tables of {n_points} points x {n_context_years} years; lower max_points.")

    def walk_forward_tables(self, eval_years: Sequence[int], max_context_rows: int = 4000, seed: int = 0):
        """Evaluation tables: for every country and eval year T, context = the country's points at every
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
