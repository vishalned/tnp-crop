"""CY-Bench's tabular features, computed from the processed WOFOST table.

A re-implementation of how AgML-CY-Bench-Modeling builds the feature table
for its tabular models (`cybench/datasets/data_factory.py`, `alignment.py`,
`conf/dataset/temporal/feature_design.yaml`,
`conf/dataset/non_temporal/geometric_transform.yaml`), so NanoTabICL sees
the same columns, in the same order and form, in pretraining and in the
benchmark. The spec, with references to the benchmark's code, is in
`docs/cybench_features.md`; `tests/test_cybench_features.py` checks this
module against the benchmark's own functions.

Per row (one season), with the crop calendar's SOS/EOS (day of year):

1. Window: from SOS - 60 days to the horizon's cutoff (`eos`: EOS;
   `middle-of-season`: EOS - season length // 2, ...), both included.
2. Derived daily series: `gdd = clip(tavg - tbase, 0, cap)` (per crop) and
   running sums (`cum_*`) from the window start.
3. 30-day blocks counted back from the cutoff: block 0 = the 30 days ending
   at the cutoff, block 1 the 30 before, ... (the earliest is usually
   partial). FPAR only has values on dekad dates (days 1, 11, 21), as JRC
   FPAR in CY-Bench.
4. One statistic (or several) per variable and block; columns
   `<variable>_<stat>_<block>`, per source sorted by name then block
   (meteo, fpar, soil_moisture), then the static features.
5. Columns with more than 5% NaN are dropped.

Left out on purpose (not in WOFOST): `vpd` (meteo), `ndvi` (source), and
`drainage_class` (static). The benchmark runs that NanoTabICL is compared in
must drop them too. The benchmark's normalizer (standardisation per variable,
pooled over blocks) is `fit_normalizer`/`apply_normalizer`.
"""

import warnings
from typing import Optional

import numpy as np
import pandas as pd

WINDOW_DAYS = 30  # `aggregate: 30`
START_OFFSET_DAYS = 60  # `start_of_sequence: sos-60`
MAX_COLUMN_NAN_RATE = 0.05  # `max_column_nan_rate`
DEKAD_DAYS = (1, 11, 21)  # JRC FPAR dates

# source -> (date resolution, created features (name, type, input), variable -> statistics);
# feature_design.yaml without vpd and the ndvi source. Sources in the benchmark's order.
SOURCES = {
    "meteo": (
        "daily",
        [("gdd", "gdd", "tavg"), ("cum_gdd", "cumsum", "gdd"), ("cum_prec", "cumsum", "prec"),
         ("cum_cwb", "cumsum", "cwb")],
        {"tmin": ["min"], "tmax": ["max"], "tavg": ["mean"], "prec": ["sum", "max"], "rad": ["mean"],
         "et0": ["mean"], "cwb": ["mean"], "gdd": ["sum"], "cum_gdd": ["max"], "cum_prec": ["max"],
         "cum_cwb": ["mean"]},
    ),
    "fpar": ("dekad", [("cum_fpar", "cumsum", "fpar")], {"fpar": ["mean"], "cum_fpar": ["max"]}),
    "soil_moisture": ("daily", [], {"ssm": ["mean"], "rsm": ["mean"]}),
}
# conf/dataset/crop/<crop>.yaml: (tbase, upper_limit)
GDD_PARAMS = {"maize": (10.0, 35.0), "wheat": (0.0, None)}
# geometric_transform.yaml without drainage_class
STATIC_FEATURES = ["sos_sin", "sos_cos", "eos_sin", "eos_cos", "loc_x", "loc_y", "loc_z", "awc", "bulk_density"]
# normalizer `fit`: standardised static features (sin/cos and loc_* are left as they are)
STANDARDIZED_STATICS = ["awc", "bulk_density"]


def season_dates(df: pd.DataFrame) -> tuple:
    """(sos_date, eos_date) per row, as the benchmark computes them
    (`compute_crop_season_window`): day of year (truncated) in the harvest
    year `year`; SOS > EOS = the season started the year before."""
    sos = df["sos_doy"].astype(float).astype(int).to_numpy()
    eos = df["eos_doy"].astype(float).astype(int).to_numpy()
    year = df["year"].astype(int).to_numpy()
    sos_date = pd.to_datetime(year * 1000 + np.clip(sos, 1, 366), format="%Y%j")
    eos_date = pd.to_datetime(year * 1000 + np.clip(eos, 1, 366), format="%Y%j")
    sos_date = pd.DatetimeIndex(np.where(sos > eos, sos_date - pd.DateOffset(years=1), sos_date))
    return sos_date.values.astype("datetime64[D]"), eos_date.values.astype("datetime64[D]")


def cutoff_days(horizon: str, season_length: np.ndarray) -> np.ndarray:
    """Days between EOS and the end of the window (`add_cutoff_days`)."""
    if horizon == "eos":
        return np.zeros_like(season_length)
    if horizon.startswith("eos-"):
        return np.full_like(season_length, int(horizon.split("-")[-1]))
    if horizon in ("early-season", "early_season"):
        return season_length * 3 // 4
    if horizon in ("middle-of-season", "mid-season"):
        return season_length // 2
    if horizon == "quarter-of-season":
        return season_length // 4
    raise ValueError(f"Unknown horizon {horizon!r}.")


def _daily(df: pd.DataFrame, name: str) -> np.ndarray:
    cols = sorted(c for c in df.columns if c.startswith(f"{name}_d") and c[len(name) + 2:].isdigit())
    if not cols:
        raise KeyError(f"No daily '{name}_dNNN' columns in the table.")
    return df[cols].to_numpy(np.float64)


def _aggregate(values: np.ndarray, groups: np.ndarray, stat: str) -> np.ndarray:
    """pandas groupby-agg semantics: NaN for a block with no observations;
    skip NaN values; an existing block of only NaN sums to 0."""
    present = groups.any(axis=1)
    x = np.where(groups, values, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = {"min": np.nanmin, "max": np.nanmax, "mean": np.nanmean, "sum": np.nansum}[stat](x, axis=1)
    return np.where(present, out, np.nan)


def cybench_features(df: pd.DataFrame, horizon: str = "eos", drop_sparse: bool = True) -> tuple:
    """Raw (unnormalised) CY-Bench features of every row of a processed WOFOST
    table (`process_wofost_dataset.py --align calendar`): a float32 array
    [rows, features] (NaN where the benchmark has NaN) and the column names."""
    crops = df["crop"].unique()
    if len(crops) != 1:
        raise ValueError(f"One crop per table, got {list(crops)}.")
    for col in ("window_start", "sos_doy", "eos_doy", "year"):
        if col not in df:
            raise KeyError(f"'{col}' missing: rebuild the table with process_wofost_dataset.py --align calendar.")
    tbase, cap = GDD_PARAMS[crops[0]]

    sos_date, eos_date = season_dates(df)
    start = sos_date - np.timedelta64(START_OFFSET_DAYS, "D")
    window_start = pd.to_datetime(df["window_start"]).values.astype("datetime64[D]")
    if (window_start > start).any():
        raise ValueError("The table's windows start after SOS - 60 days; rebuild it with --align calendar.")
    season_length = (eos_date - sos_date).astype(int)
    end = eos_date - cutoff_days(horizon, season_length).astype("timedelta64[D]")

    tavg = _daily(df, "tavg")
    n_days = tavg.shape[1]
    dates = window_start[:, None] + np.arange(n_days).astype("timedelta64[D]")[None, :]
    in_window = (dates >= start[:, None]) & (dates <= end[:, None])
    block = (end[:, None] - dates).astype(int) // WINDOW_DAYS
    day_of_month = (dates - dates.astype("datetime64[M]")).astype(int) + 1
    on_dates = {"daily": in_window, "dekad": in_window & np.isin(day_of_month, DEKAD_DAYS)}

    blocks, names = [], []
    for resolution, created, stats in SOURCES.values():
        observed = on_dates[resolution]  # the source's rows inside the window
        series = {}
        for var in stats:
            if not any(var == c[0] for c in created):
                series[var] = np.where(observed, _daily(df, var), np.nan)
        for name, kind, source in created:
            if kind == "gdd":
                series[name] = np.clip(series[source] - tbase, 0.0, cap)
            elif kind == "cumsum":  # running total from the window start, skipping NaN
                series[name] = np.where(np.isnan(series[source]), np.nan,
                                        np.nancumsum(np.where(observed, series[source], np.nan), axis=1))
        n_blocks = int(block[observed].max()) + 1
        columns = {}
        for var, var_stats in stats.items():
            for stat in var_stats:
                columns[f"{var}_{stat}"] = [_aggregate(series[var], observed & (block == b), stat)
                                            for b in range(n_blocks)]
        for name in sorted(columns):  # pivot_table sorts by feature name, then block
            for b, values in enumerate(columns[name]):
                blocks.append(values)
                names.append(f"{name}_{b}")

    sos_doy, eos_doy = df["sos_doy"].to_numpy(float), df["eos_doy"].to_numpy(float)
    lat, lon = np.deg2rad(df["latitude"].to_numpy(float)), np.deg2rad(df["longitude"].to_numpy(float))
    statics = {
        "sos_sin": np.sin(2 * np.pi * sos_doy / 365), "sos_cos": np.cos(2 * np.pi * sos_doy / 365),
        "eos_sin": np.sin(2 * np.pi * eos_doy / 365), "eos_cos": np.cos(2 * np.pi * eos_doy / 365),
        "loc_x": np.cos(lat) * np.cos(lon), "loc_y": np.cos(lat) * np.sin(lon), "loc_z": np.sin(lat),
        "awc": df["awc"].to_numpy(float), "bulk_density": df["bulk_density"].to_numpy(float),
    }
    for name in STATIC_FEATURES:
        blocks.append(statics[name])
        names.append(name)

    features = np.stack(blocks, axis=1)
    if drop_sparse:
        keep = np.isnan(features).mean(axis=0) <= MAX_COLUMN_NAN_RATE
        features, names = features[:, keep], [n for n, k in zip(names, keep) if k]
    return features.astype(np.float32), names


def variable_of(name: str) -> str:
    """The normaliser's feature name of a column: `tmin_min_3` -> `tmin_min`."""
    head, _, tail = name.rpartition("_")
    return head if tail.isdigit() else name


def fit_normalizer(features: np.ndarray, names: list, rows: Optional[np.ndarray] = None) -> dict:
    """The benchmark's `fit` normaliser: mean/std (ddof 1) per aggregated
    variable, pooled over its blocks, and per standardised static feature;
    fitted on `rows` (e.g. the train years). {variable: (mean, std)}."""
    x = features if rows is None else features[rows]
    params = {}
    for var in dict.fromkeys(variable_of(n) for n in names):
        if var in STATIC_FEATURES and var not in STANDARDIZED_STATICS:
            continue
        values = x[:, [variable_of(n) == var for n in names]].astype(np.float64).ravel()
        values = values[~np.isnan(values)]
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        params[var] = (float(values.mean()) if len(values) else 0.0, std if np.isfinite(std) else 0.0)
    return params


def apply_normalizer(features: np.ndarray, names: list, params: dict) -> np.ndarray:
    """(x - mean) / std per column (0 where std is 0), as the benchmark's
    Normalizer; columns without parameters are left as they are."""
    out = features.astype(np.float32, copy=True)
    for j, name in enumerate(names):
        if variable_of(name) in params:
            mean, std = params[variable_of(name)]
            out[:, j] = (out[:, j] - mean) / std if std > 0 else out[:, j] * 0.0
    return out
