"""Episode sampling, aggregation, normalization and tokenization for the
crop TNP, on top of the training store built by
`src/data_pipeline/wofost/build_training_store.py`.

One episode (in this order, see `EpisodeSampler.sample`):

1. country ~ uniform over the configured countries (not weighted by point
   count);
2. X ~ U{min_points..max_points} points in that country;
3. one shared target season year T, uniform over the split's target years
   that leave >= `min_context_years` earlier years in the context pool;
4. one shared context length C ~ U{min_context_years..years available
   before T} (optionally capped by `max_context_years`);
5. per point, C distinct context years drawn from its years before T;
6. per (point, year) cell, one of its jitter runs, chosen uniformly -- never
   more than one per cell;
7. one temporal density profile for the whole episode (weekly, dekadal or
   irregular-gappy), used to aggregate every cell's daily weather with the
   per-variable function (mean for state variables, sum for fluxes);
8. every value z-scored with train-pool-only statistics.

Tokens are `(coordinate [lat, lon, t, depth], modality_id, value)`:
- context: static tokens per point (once), weather tokens for every cell
  (context years and the target year alike), and label tokens (yield,
  phenology) for the context years, values shown;
- target: the label tokens of every point at year T, values to predict.

Label tokens are placed at the (nominal) season start of their season, not
at the event date: a target token's coordinate is visible to the model, so
placing e.g. the maturity query at the true maturity date would leak the
answer. Phenology values are days after that season start.
"""

import datetime
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from src.data.components.crop_vocab import (
    LABEL_COLUMNS,
    LAYERED_STATIC_VARIABLES,
    MODALITY_ID,
    SOIL_LAYER_MID_DEPTHS_CM,
    TIME_REFERENCE_DATE,
    WEATHER_AGGREGATION,
    WEATHER_VARIABLES,
)

REFERENCE_DATE = datetime.date.fromisoformat(TIME_REFERENCE_DATE)
PROFILES = ("weekly", "dekadal", "irregular")


@dataclass
class EpisodeConfig:
    countries: Sequence[str]
    min_points: int = 3
    max_points: int = 10
    min_context_years: int = 5
    max_context_years: Optional[int] = None  # None = all years available before T
    profiles: Sequence[str] = PROFILES
    static_variables: Sequence[str] = ("clay", "nitrogen", "ph", "soc", "water_holding_capacity", "elevation", "slope")
    label_modalities: Sequence[str] = ("yield", "phenology_maturity")
    window_days: int = 322  # weather window per season, from season start - pre_season_days
    pre_season_days: int = 14
    weekly_days: int = 7
    dekadal_days: int = 10
    irregular_bucket_days: int = 7  # fixed interval length, so sums stay comparable
    irregular_max_gap_days: int = 10  # random gap (0..max) before each interval
    require_maturity: bool = True  # drop seasons that never reached maturity

    def bucket_days(self, profile: str) -> int:
        return {"weekly": self.weekly_days, "dekadal": self.dekadal_days, "irregular": self.irregular_bucket_days}[profile]


class CropStore:
    """The training store in memory, indexed for episode sampling."""

    def __init__(self, store_dir: str, config: EpisodeConfig):
        self.store_dir = store_dir
        self.config = config
        self.points = pd.read_csv(os.path.join(store_dir, "points.csv"))
        if not (self.points["point_id"].to_numpy() == np.arange(len(self.points))).all():
            raise ValueError("points.csv must list point_id 0..N-1 in order (as build_training_store writes it).")
        self.seasons = pd.read_csv(os.path.join(store_dir, "seasons.csv"), parse_dates=["season_start"])
        # memory-mapped: shared between DataLoader workers through the page cache
        self.weather = np.load(os.path.join(store_dir, "weather.npy"), mmap_mode="r")
        with open(os.path.join(store_dir, "weather_meta.json")) as f:
            meta = json.load(f)
        if meta["variables"] != WEATHER_VARIABLES:
            raise ValueError(f"Store weather variables {meta['variables']} != expected {WEATHER_VARIABLES}.")
        self.weather_start = datetime.date.fromisoformat(meta["start_date"])
        self.axis_offset = (self.weather_start - REFERENCE_DATE).days  # t of weather day index 0

        self.lat = self.points["latitude"].to_numpy(dtype=np.float32)
        self.lon = self.points["longitude"].to_numpy(dtype=np.float32)

        # A season is usable if it has every configured label (and matured, if required).
        label_cols = [LABEL_COLUMNS[m] for m in config.label_modalities]
        ok = self.seasons[label_cols].notna().all(axis=1)
        if config.require_maturity:
            ok &= self.seasons["reached_maturity"].astype(bool)
        usable = self.seasons[ok]

        # (point, season_year) -> season row indices (its jitters); season start day index
        self.cell_jitters: Dict[tuple, np.ndarray] = {
            key: rows.index.to_numpy() for key, rows in usable.groupby(["point_id", "season_year"])
        }
        starts = self.seasons.groupby(["point_id", "season_year"])["season_start"].first()
        self.cell_start_index = {
            key: (ts.date() - self.weather_start).days for key, ts in starts.items()
        }
        self.point_years: Dict[int, np.ndarray] = {}
        for (p, y) in self.cell_jitters:
            self.point_years.setdefault(p, []).append(y)
        self.point_years = {p: np.array(sorted(ys)) for p, ys in self.point_years.items()}
        self.years = np.array(sorted(usable["season_year"].unique()))

        countries = self.points.set_index("point_id")["country"]
        self.country_points = {
            c: np.array([p for p in countries.index[countries == c] if p in self.point_years])
            for c in config.countries
        }
        missing = [c for c, pts in self.country_points.items() if len(pts) == 0]
        if missing:
            print(f"Warning: no usable points for countries {missing}; they are left out of episode sampling.")
        self.countries = [c for c, pts in self.country_points.items() if len(pts) > 0]
        if not self.countries:
            raise ValueError(f"None of the countries {list(config.countries)} have usable points in {store_dir}.")

    def static_columns(self) -> List[tuple]:
        """(modality name, store column, depth cm) for every static token of a point."""
        cols = []
        for name in self.config.static_variables:
            if name in LAYERED_STATIC_VARIABLES:
                layer = 0
                while f"{name}_{layer}" in self.points:
                    cols.append((name, f"{name}_{layer}", SOIL_LAYER_MID_DEPTHS_CM[layer]))
                    layer += 1
            elif name in self.points:
                cols.append((name, name, 0.0))
            else:
                raise ValueError(f"Static variable '{name}' is not in the store's points.csv.")
        return cols

    def window(self, point_id: int, season_year: int) -> np.ndarray:
        """Daily weather [window_days, num_vars] of one season (NaN where unavailable)."""
        cfg = self.config
        i0 = self.cell_start_index[(point_id, season_year)] - cfg.pre_season_days
        out = np.full((cfg.window_days, len(WEATHER_VARIABLES)), np.nan, dtype=np.float32)
        lo, hi = max(i0, 0), min(i0 + cfg.window_days, self.weather.shape[1])
        if hi > lo:
            out[lo - i0 : hi - i0] = self.weather[point_id, lo:hi]
        return out


def intervals_for_profile(profile: str, config: EpisodeConfig, rng: np.random.Generator) -> np.ndarray:
    """[n, 2] (start offset, length) of the buckets within one season window."""
    L = config.bucket_days(profile)
    if profile in ("weekly", "dekadal"):
        n = config.window_days // L
        return np.stack([np.arange(n) * L, np.full(n, L)], axis=1)
    starts, pos = [], 0
    while True:
        pos += int(rng.integers(0, config.irregular_max_gap_days + 1))
        if pos + L > config.window_days:
            break
        starts.append(pos)
        pos += L
    return np.stack([np.array(starts, dtype=int), np.full(len(starts), L)], axis=1).reshape(-1, 2)


_SUM_MASK = np.array([WEATHER_AGGREGATION[v] == "sum" for v in WEATHER_VARIABLES])


def aggregate(daily: np.ndarray, intervals: np.ndarray) -> np.ndarray:
    """[n, num_vars] bucket values: sum for flux variables, mean otherwise.
    A bucket with any missing day is NaN (a partial sum would be biased)."""
    if len(intervals) == 0:
        return np.zeros((0, daily.shape[1]), dtype=np.float32)
    L = int(intervals[0, 1])
    idx = intervals[:, :1] + np.arange(L)[None, :]  # [n, L]
    chunk = daily[idx]  # [n, L, V]
    summed = chunk.sum(axis=1)  # NaN propagates
    return np.where(_SUM_MASK[None, :], summed, summed / L).astype(np.float32)


def compute_norm_stats(store: CropStore, train_years: Sequence[int]) -> dict:
    """Mean/std of every token variable, from the train-pool years only.

    Weather stats are per bucket length (a weekly precipitation sum and a
    dekadal one have different scales); static stats pool all layers of a
    variable; label stats use every usable train-pool season (all jitters).
    """
    cfg = store.config
    train_years = set(int(y) for y in train_years)
    lengths = sorted({cfg.bucket_days(p) for p in cfg.profiles})

    weather = {}
    cells = [key for key in store.cell_start_index if key[1] in train_years and key[0] in store.point_years]
    for L in lengths:
        intervals = np.stack([np.arange(cfg.window_days // L) * L, np.full(cfg.window_days // L, L)], axis=1)
        values = np.concatenate([aggregate(store.window(p, y), intervals) for p, y in cells], axis=0)
        weather[str(L)] = {v: _mean_std(values[:, i]) for i, v in enumerate(WEATHER_VARIABLES)}

    static = {}
    for name, col, _ in store.static_columns():
        static.setdefault(name, []).append(store.points[col].to_numpy(dtype=np.float64))
    static = {name: _mean_std(np.concatenate(v)) for name, v in static.items()}

    in_train = store.seasons["season_year"].isin(train_years)
    labels = {m: _mean_std(store.seasons.loc[in_train, col].to_numpy(dtype=np.float64)) for m, col in LABEL_COLUMNS.items()}

    return {
        "train_years": sorted(train_years),
        "window_days": cfg.window_days,
        "pre_season_days": cfg.pre_season_days,
        "weather": weather,
        "static": static,
        "labels": labels,
    }


def _mean_std(x: np.ndarray) -> List[float]:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return [0.0, 1.0]
    std = float(x.std())
    return [float(x.mean()), std if std > 1e-8 else 1.0]


@dataclass
class EpisodeInfo:
    country: str
    target_year: int
    num_points: int
    context_years: int
    profile: str
    num_context_tokens: int = 0
    num_target_tokens: int = 0
    point_ids: list = field(default_factory=list)


class EpisodeSampler:
    """Samples and tokenizes episodes for one split.

    :param target_years: years T may be drawn from.
    :param context_pool_years: years context may be drawn from (those < T).
        Train: the train pool. Val/test (walk-forward): every year before T.
    """

    def __init__(self, store: CropStore, stats: dict, target_years: Sequence[int], context_pool_years: Sequence[int]):
        self.store = store
        self.stats = stats
        self.cfg = store.config
        self.context_pool = np.array(sorted(context_pool_years))
        self.target_years = [
            int(t) for t in target_years
            if (self.context_pool < t).sum() >= self.cfg.min_context_years and t in set(store.years)
        ]
        if not self.target_years:
            raise ValueError(
                f"No target year in {list(target_years)} has >= {self.cfg.min_context_years} earlier context years "
                f"in {list(context_pool_years)} with data."
            )
        self.static_cols = store.static_columns()

    def sample(self, rng: np.random.Generator, max_attempts: int = 100) -> tuple:
        cfg, store = self.cfg, self.store
        for _ in range(max_attempts):
            country = store.countries[rng.integers(len(store.countries))]  # 1. uniform over countries
            num_points = int(rng.integers(cfg.min_points, cfg.max_points + 1))  # 2.
            T = int(rng.choice(self.target_years))  # 3.
            pool_before = self.context_pool[self.context_pool < T]
            max_c = len(pool_before) if cfg.max_context_years is None else min(len(pool_before), cfg.max_context_years)
            C = int(rng.integers(cfg.min_context_years, max_c + 1))  # 4.

            # points of this country with a usable target cell and >= C usable context years
            candidates = []
            for p in store.country_points[country]:
                years = store.point_years[p]
                if T in years and np.isin(years, pool_before).sum() >= C:
                    candidates.append(p)
            if not candidates:
                continue
            chosen = rng.choice(candidates, size=min(num_points, len(candidates)), replace=False)
            profile = cfg.profiles[rng.integers(len(cfg.profiles))]  # 7.
            info = EpisodeInfo(country, T, len(chosen), C, profile, point_ids=[int(p) for p in chosen])
            return self._tokenize(rng, chosen, T, C, pool_before, profile, info)
        raise RuntimeError(f"Couldn't sample a valid episode in {max_attempts} attempts; check the store/config.")

    def _tokenize(self, rng, points, T, C, pool_before, profile, info) -> tuple:
        cfg, store, stats = self.cfg, self.store, self.stats
        L = cfg.bucket_days(profile)
        wstats = stats["weather"][str(L)]
        w_mean = np.array([wstats[v][0] for v in WEATHER_VARIABLES], dtype=np.float32)
        w_std = np.array([wstats[v][1] for v in WEATHER_VARIABLES], dtype=np.float32)
        w_ids = np.array([MODALITY_ID[v] for v in WEATHER_VARIABLES])

        ctx_x, ctx_y, ctx_m, tar_x, tar_y, tar_m = [], [], [], [], [], []

        def add(xs, ys, ms, lat, lon, t, depth, value, mod):
            n = len(value)
            xs.append(np.stack([np.full(n, lat), np.full(n, lon), t, depth], axis=1).astype(np.float32))
            ys.append(np.asarray(value, dtype=np.float32).reshape(-1, 1))
            ms.append(np.asarray(mod, dtype=np.int64))

        for p in points:
            lat, lon = store.lat[p], store.lon[p]

            # static tokens, once per point
            names, cols, depths = zip(*self.static_cols)
            raw = store.points.loc[p, list(cols)].to_numpy(dtype=np.float32)
            norm = np.array([(r - stats["static"][n][0]) / stats["static"][n][1] for r, n in zip(raw, names)], dtype=np.float32)
            keep = np.isfinite(norm)
            add(ctx_x, ctx_y, ctx_m, lat, lon, np.zeros(keep.sum()), np.array(depths)[keep], norm[keep],
                [MODALITY_ID[n] for n, k in zip(names, keep) if k])

            years = store.point_years[p]
            context_years = rng.choice(years[np.isin(years, pool_before)], size=C, replace=False)  # 5.
            for y in list(context_years) + [T]:
                is_target = y == T
                season_row = rng.choice(store.cell_jitters[(p, y)])  # 6. one jitter per cell
                season_t = float(store.cell_start_index[(p, y)] + store.axis_offset)

                # weather tokens (always context), aggregated to this episode's profile (7.) and z-scored (8.)
                intervals = intervals_for_profile(profile, cfg, rng)
                values = (aggregate(store.window(p, y), intervals) - w_mean) / w_std  # [n, V]
                t_bucket = (
                    store.cell_start_index[(p, y)] - cfg.pre_season_days + store.axis_offset
                    + intervals[:, 0] + (L - 1) / 2.0
                )
                t_tok = np.repeat(t_bucket, len(WEATHER_VARIABLES))
                v_tok = values.reshape(-1)
                m_tok = np.tile(w_ids, len(intervals))
                keep = np.isfinite(v_tok)
                add(ctx_x, ctx_y, ctx_m, lat, lon, t_tok[keep], np.zeros(keep.sum()), v_tok[keep], m_tok[keep])

                # label tokens: shown for context years, queried for T
                labels = [
                    (store.seasons.at[season_row, LABEL_COLUMNS[m]] - stats["labels"][m][0]) / stats["labels"][m][1]
                    for m in cfg.label_modalities
                ]
                label_ids = [MODALITY_ID[m] for m in cfg.label_modalities]
                n = len(labels)
                if is_target:
                    add(tar_x, tar_y, tar_m, lat, lon, np.full(n, season_t), np.zeros(n), labels, label_ids)
                else:
                    add(ctx_x, ctx_y, ctx_m, lat, lon, np.full(n, season_t), np.zeros(n), labels, label_ids)

        episode = {
            "xc": torch.from_numpy(np.concatenate(ctx_x)),
            "yc": torch.from_numpy(np.concatenate(ctx_y)),
            "mc": torch.from_numpy(np.concatenate(ctx_m)),
            "xt": torch.from_numpy(np.concatenate(tar_x)),
            "yt": torch.from_numpy(np.concatenate(tar_y)),
            "mt": torch.from_numpy(np.concatenate(tar_m)),
        }
        info.num_context_tokens = len(episode["xc"])
        info.num_target_tokens = len(episode["xt"])
        return episode, info


class TrainEpisodeDataset(IterableDataset):
    """Fresh random episodes every epoch (`episodes_per_epoch` per epoch,
    split across DataLoader workers)."""

    def __init__(self, sampler: EpisodeSampler, episodes_per_epoch: int, seed: int = 0):
        super().__init__()
        self.sampler = sampler
        self.episodes_per_epoch = episodes_per_epoch
        self.seed = seed
        self._iterations = 0

    def __len__(self) -> int:
        return self.episodes_per_epoch

    def __iter__(self):
        worker = get_worker_info()
        worker_id, num_workers = (worker.id, worker.num_workers) if worker else (0, 1)
        # Workers are re-created (with a fresh torch seed) every epoch; in the
        # main process the iteration counter changes the stream instead.
        epoch_entropy = torch.initial_seed() % 2**32 if worker else self._iterations
        self._iterations += 1
        rng = np.random.default_rng([self.seed, epoch_entropy, worker_id])
        n = self.episodes_per_epoch // num_workers + (worker_id < self.episodes_per_epoch % num_workers)
        for _ in range(n):
            episode, info = self.sampler.sample(rng)
            episode["info"] = asdict(info)
            yield episode


class EvalEpisodeDataset(Dataset):
    """A fixed set of walk-forward episodes (the same every epoch): episode
    `i` always comes from the same seed."""

    def __init__(self, sampler: EpisodeSampler, num_episodes: int, seed: int):
        self.sampler = sampler
        self.num_episodes = num_episodes
        self.seed = seed

    def __len__(self) -> int:
        return self.num_episodes

    def __getitem__(self, i: int) -> dict:
        episode, info = self.sampler.sample(np.random.default_rng([self.seed, i]))
        episode["info"] = asdict(info)
        return episode


def collate_episodes(episodes: List[dict]) -> dict:
    """Pad a list of variable-length episodes into one batch.

    Context and target sets are padded separately, to the batch's longest
    context and longest target set. `mask_c`/`mask_t` are True for real
    tokens: the model blocks attention to padded context positions and the
    loss ignores padded target positions.
    """
    B = len(episodes)
    nc = max(len(e["xc"]) for e in episodes)
    nt = max(len(e["xt"]) for e in episodes)
    out = {
        "xc": torch.zeros(B, nc, 4), "yc": torch.zeros(B, nc, 1), "mc": torch.zeros(B, nc, dtype=torch.long),
        "mask_c": torch.zeros(B, nc, dtype=torch.bool),
        "xt": torch.zeros(B, nt, 4), "yt": torch.zeros(B, nt, 1), "mt": torch.zeros(B, nt, dtype=torch.long),
        "mask_t": torch.zeros(B, nt, dtype=torch.bool),
    }
    for b, e in enumerate(episodes):
        for s, n in (("c", len(e["xc"])), ("t", len(e["xt"]))):
            out[f"x{s}"][b, :n] = e[f"x{s}"]
            out[f"y{s}"][b, :n] = e[f"y{s}"]
            out[f"m{s}"][b, :n] = e[f"m{s}"]
            out[f"mask_{s}"][b, :n] = True
    out["num_ctx"] = out["mask_c"].sum(1)
    out["num_tar"] = out["mask_t"].sum(1)
    out["info"] = [e.get("info") for e in episodes]
    return out
