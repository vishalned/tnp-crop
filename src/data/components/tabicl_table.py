"""Flatten the WOFOST training store (`CropStore`, from
`build_training_store.py`) into the tabular shape TabICL expects: a plain
``(n_samples, n_features)`` feature matrix and a ``(n_samples,)`` target
vector -- no episodes, no tokens.

TabICL does its own context/query splitting and ensembling internally
(`tabicl._finetune.data.iter_epoch_meta_batches`, driven by
`FinetunedTabICLRegressor.fit`), so unlike the TNP-D pipeline
(`crop_episode_dataset.py`) this module's only job is to produce one row
per usable (point, season_year, jitter) cell:

- static soil/terrain columns, same ones and names as `CropStore.points`
  (`clay_0..2`, `nitrogen_0..2`, ..., `water_holding_capacity`, `elevation`,
  `slope`);
- weather columns: each season's daily window aggregated into fixed buckets
  (reusing the TNP tokenizer's own `aggregate`/`intervals_for_profile`, at a
  single fixed profile instead of one sampled per episode) and flattened to
  `{variable}_b{k:02d}` columns, one per (bucket, variable);
- id/group columns (`point_id`, `country`, `season_year`, `jitter_index`,
  `latitude`, `longitude`) for splitting and bookkeeping -- drop these
  before handing the table to TabICL (see `table_to_xy`);
- one column per configured label modality (`yield_t_per_ha`,
  `maturity_days`, ...), un-normalized (TabICL fits its own preprocessing).

A coarser `profile` (e.g. "dekadal") trades weather resolution for fewer
feature columns.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.data.components.crop_episode_dataset import CropStore, aggregate, intervals_for_profile
from src.data.components.crop_vocab import LABEL_COLUMNS, WEATHER_VARIABLES

ID_COLUMNS = ["point_id", "country", "latitude", "longitude", "season_year", "jitter_index"]


@dataclass
class TabICLTableConfig:
    """What goes into a table row, on top of the store's own `EpisodeConfig`
    (which fixes which (point, season_year) cells are usable at all, via its
    `label_modalities`/`require_maturity`)."""

    profile: str = "weekly"  # bucket resolution for the flattened weather columns
    label_modalities: Sequence[str] = ("yield", "phenology_maturity")


def _weather_columns(store: CropStore, profile: str) -> Tuple[np.ndarray, List[str]]:
    """[n_buckets, 2] (start offset, length) intervals and their flattened
    column names `{variable}_b{k:02d}`, for the store's fixed window."""
    cfg = store.config
    intervals = intervals_for_profile(profile, cfg, np.random.default_rng(0))  # weekly/dekadal: deterministic
    names = [f"{v}_b{k:02d}" for k in range(len(intervals)) for v in WEATHER_VARIABLES]
    return intervals, names


def build_table(store: CropStore, config: Optional[TabICLTableConfig] = None) -> pd.DataFrame:
    """One row per usable (point, season_year, jitter) cell of `store`.

    Mirrors the row set `EpisodeSampler` can draw from (`store.cell_jitters`,
    already filtered by the store's `EpisodeConfig.label_modalities` /
    `require_maturity`), just without episode sampling: every usable cell
    becomes a row instead of only the ones a sampled episode happens to use.
    """
    config = config or TabICLTableConfig()
    if config.profile == "irregular":
        raise ValueError(
            "profile='irregular' has a random bucket count/placement per draw; every row here must share the "
            "same weather columns, so use 'weekly' or 'dekadal' (fixed, regular buckets) instead."
        )
    intervals, weather_cols = _weather_columns(store, config.profile)
    static_cols = store.static_columns()  # [(name, column, depth), ...]
    static_col_names = [col for _, col, _ in static_cols]
    label_cols = {m: LABEL_COLUMNS[m] for m in config.label_modalities}

    rows = []
    for (point_id, season_year), row_idxs in store.cell_jitters.items():
        lat, lon = float(store.lat[point_id]), float(store.lon[point_id])
        country = store.points.at[point_id, "country"]
        static_values = store.points.loc[point_id, static_col_names].to_numpy(dtype=np.float32)
        weather = aggregate(store.window(point_id, season_year), intervals).reshape(-1)  # [n_buckets * V]

        for row_idx in row_idxs:
            season = store.seasons.loc[row_idx]
            row = {
                "point_id": point_id,
                "country": country,
                "latitude": lat,
                "longitude": lon,
                "season_year": int(season_year),
                "jitter_index": int(season["jitter_index"]),
            }
            row.update(zip(static_col_names, static_values))
            row.update(zip(weather_cols, weather))
            for modality, col in label_cols.items():
                row[col] = season[col]
            rows.append(row)

    columns = ID_COLUMNS + static_col_names + weather_cols + list(label_cols.values())
    return pd.DataFrame(rows, columns=columns)


def split_table(
    table: pd.DataFrame,
    train_years: Sequence[int] = (2005, 2016),
    val_years: Sequence[int] = (2017, 2018),
    test_years: Sequence[int] = (2019, 2020),
) -> Dict[str, pd.DataFrame]:
    """Split by `season_year` (sowing year), the same split points as
    `CropEpisodeDataModule`. Unlike that datamodule's walk-forward
    val/test (context = any earlier year, picked per episode), here every
    split is just its own fixed slice of rows -- TabICL draws its own
    context/query split from whatever table it is given."""

    def _between(bounds):
        lo, hi = bounds
        return table[(table["season_year"] >= lo) & (table["season_year"] <= hi)]

    return {"train": _between(train_years), "val": _between(val_years), "test": _between(test_years)}


def table_to_xy(table: pd.DataFrame, target: str, id_columns: Sequence[str] = ID_COLUMNS) -> Tuple[pd.DataFrame, pd.Series]:
    """Drop id/group columns and every label column except `target`
    (TabICL fits one target at a time) -> `(X, y)` ready for
    `FinetunedTabICLRegressor.fit(X, y)` / `.predict(X)`."""
    other_labels = [c for c in LABEL_COLUMNS.values() if c in table.columns and c != target]
    X = table.drop(columns=[*id_columns, *other_labels, target])
    y = table[target]
    return X, y
