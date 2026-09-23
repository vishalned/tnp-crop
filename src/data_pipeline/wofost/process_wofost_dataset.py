"""Turn a WOFOST batch run (its `dataset_manifest.csv` plus the per-episode
daily CSVs and per-location weather caches under data/raw) into one
training table per crop under data/processed.

One row = one simulated growing season (sowing -> maturity) at one
location, the same unit as a CYBench row (region x season). Rows are
labelled with the harvest year (the year the crop matured), CYBench's
convention: winter wheat sown in Oct 2013 and harvested in Jul 2014 is
year 2014. The sowing year is kept as `sowing_year`.

Time series are aggregated into fixed-length buckets (weekly by default)
counted from the nominal season start, so every row of a crop has the same
columns:

- `align="season_start"` (default): bucket 0 starts on the crop's nominal
  start of season for that year (`default_sowing_doy()`, the stand-in for
  the WorldCereal SOS that CYBench also aligns to). The actual, jittered
  sowing date then shows up inside the series, as it would in real data.
- `align="sowing"`: bucket 0 starts on the actual sowing date.

The window covers `default_max_duration_days()` (the longest a season can
run), so it doesn't depend on when the crop matured. Weather for the whole
window comes from the per-location weather cache, so it is complete even
after maturity. Simulated crop/soil variables only exist while the crop is
growing: `fpar` is 0 outside it (bare field before sowing / after harvest),
`ssm` is left NaN there (it isn't simulated).

Time-series column names are `{feature}_{prefix}{bucket:02d}`, e.g.
`tmax_w00` for the first week. The exact columns of each group are written
to a `..._columns.json` sidecar next to the table, for the dataloader.
"""

import argparse
import datetime
import json
import math
import os
import sys
from typing import Optional, Union

import numpy as np
import pandas as pd
import rootutils

from src.data_pipeline.weather.utils_weather.gee_weather import (
    DEFAULT_WEATHER_CACHE_DIR,
    _weather_cache_path,
)
from src.data_pipeline.wofost.run_wofost_simulation import DEFAULT_WOFOST_SAVE_DIR
from src.data_pipeline.wofost.utils_wofost.default_wofost_variables import (
    default_max_duration_days,
    default_sowing_doy,
)


root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
DEFAULT_PROCESSED_DIR = os.path.join(str(root), "data", "processed")


def default_aggregations() -> dict:
    """Named bucket lengths (days) and the column prefix each one uses.
    `monthly` is 30-day blocks from the season start, not calendar months."""
    return {
        "daily": (1, "d"),
        "weekly": (7, "w"),
        "dekadal": (10, "dk"),
        "biweekly": (14, "bw"),
        "monthly": (30, "m"),
    }


def default_timeseries_features() -> dict:
    """CYBench-aligned time-series features: output name -> (source column,
    source, scale factor). Every feature is the bucket mean of the daily
    values (so buckets of any length, or with a missing day, stay
    comparable), after scaling to the unit given here.

    Weather comes from the per-location weather cache (PCSE units), the
    simulated features from the episode's daily WOFOST CSV.
    """
    return {
        "tmin": ("TMIN", "weather", 1.0),  # degC
        "tmax": ("TMAX", "weather", 1.0),  # degC
        "tavg": ("TEMP", "weather", 1.0),  # degC
        "prec": ("RAIN", "weather", 10.0),  # cm/day -> mm/day
        "rad": ("IRRAD", "weather", 1e-6),  # J/m2/day -> MJ/m2/day
        "et0": ("ET0", "weather", 10.0),  # cm/day -> mm/day
        "cwb": ("CWB", "weather", 10.0),  # cm/day -> mm/day (RAIN - ET0)
        "fpar": ("fpar", "wofost", 1.0),  # fraction
        "ssm": ("ssm", "wofost", 1.0),  # topsoil volumetric moisture, cm3/cm3
    }


def default_static_features() -> list:
    """Per-row scalar features (CYBench's soil + crop calendar analogues)."""
    return ["latitude", "longitude", "awc", "bulk_density", "sos_doy", "sowing_doy"]


def _resolve_aggregation(aggregation: Union[str, int]) -> tuple:
    named = default_aggregations()
    if isinstance(aggregation, str) and aggregation in named:
        return named[aggregation]
    try:
        days = int(aggregation)
    except (TypeError, ValueError):
        raise ValueError(f"aggregation must be one of {list(named)} or a number of days, got {aggregation!r}.")
    if days < 1:
        raise ValueError(f"aggregation must be at least 1 day, got {days}.")
    return days, "b"


def _resolve_path(path: str, fallback_dir: str) -> str:
    """Manifest paths are absolute on the machine that ran the batch; if the
    data has moved, fall back to the same file name under `fallback_dir`."""
    if isinstance(path, str) and os.path.exists(path):
        return path
    candidate = os.path.join(fallback_dir, os.path.basename(str(path)))
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(f"Episode file not found: {path} (also tried {candidate}).")


def _load_weather(longitude: float, latitude: float, weather_dir: str) -> pd.DataFrame:
    path = _weather_cache_path(longitude, latitude, weather_dir)
    df = pd.read_csv(path)
    df["DAY"] = pd.to_datetime(df["DAY"]).dt.date
    if "TEMP" not in df:
        # optional in PCSE records; PCSE itself falls back to this
        df["TEMP"] = (df["TMIN"] + df["TMAX"]) / 2.0
    df["CWB"] = df["RAIN"] - df["ET0"]
    return df.set_index("DAY")


def _bucket_means(daily: pd.DataFrame, window_start: datetime.date, num_buckets: int, bucket_days: int) -> pd.DataFrame:
    """Mean of each column per bucket, over [window_start, window_start + num_buckets * bucket_days)."""
    days = pd.date_range(window_start, periods=num_buckets * bucket_days, freq="D").date
    window = daily.reindex(days)
    window["bucket"] = np.repeat(np.arange(num_buckets), bucket_days)
    return window.groupby("bucket").mean()


def process_episode(
    row: pd.Series,
    weather: pd.DataFrame,
    wofost_dir: str,
    bucket_days: int,
    prefix: str,
    align: str,
    features: dict,
) -> dict:
    """One processed row (dict) for one successful manifest episode."""
    crop = row["crop"]
    daily_path = _resolve_path(row["daily_path"], os.path.join(wofost_dir, crop))
    daily = pd.read_csv(daily_path)
    daily["day"] = pd.to_datetime(daily["day"]).dt.date
    daily = daily.set_index("day")

    sowing_date = datetime.date.fromisoformat(str(row["sowing_date"]))
    sowing_year = int(row["year"])
    sos_doy = default_sowing_doy()[crop]
    season_start = datetime.date(sowing_year, 1, 1) + datetime.timedelta(days=sos_doy - 1)
    window_start = season_start if align == "season_start" else sowing_date
    num_buckets = math.ceil(default_max_duration_days()[crop] / bucket_days)

    matured = daily.index[daily["DVS"] >= 2.0]
    reached_maturity = len(matured) > 0
    maturity_date = matured[0] if reached_maturity else daily.index[-1]

    weather_cols = {name: col for name, (col, source, _) in features.items() if source == "weather"}
    wofost_cols = {name: col for name, (col, source, _) in features.items() if source == "wofost"}

    weather_buckets = _bucket_means(weather[list(weather_cols.values())], window_start, num_buckets, bucket_days)
    wofost_daily = daily[list(wofost_cols.values())].copy()
    if "fpar" in wofost_daily:
        # No green canopy outside the simulated crop life: before sowing and
        # after maturity (harvest) the field is bare.
        full_days = pd.date_range(window_start, periods=num_buckets * bucket_days, freq="D").date
        wofost_daily = wofost_daily.reindex(full_days)
        wofost_daily["fpar"] = wofost_daily["fpar"].fillna(0.0)
    wofost_buckets = _bucket_means(wofost_daily, window_start, num_buckets, bucket_days)

    out = {
        "sample_id": f"{row['longitude']}_{row['latitude']}_{crop}_{maturity_date.year}",
        "location_index": int(row["location_index"]),
        "location_id": f"{row['longitude']}_{row['latitude']}",
        "crop": crop,
        "year": maturity_date.year,
        "sowing_year": sowing_year,
        "latitude": row["latitude"],
        "longitude": row["longitude"],
        "awc": row["awc"],
        "bulk_density": row["bulk_density"],
        "sos_doy": sos_doy,
        "sowing_doy": sowing_date.timetuple().tm_yday,
        "sowing_date": sowing_date.isoformat(),
        "maturity_date": maturity_date.isoformat(),
        "season_length_days": (maturity_date - sowing_date).days,
        "reached_maturity": reached_maturity,
        "window_start": window_start.isoformat(),
        "yield_kg_per_ha": row["yield_kg_per_ha"],
        "yield_t_per_ha": row["yield_kg_per_ha"] / 1000.0,
    }
    for name, (col, source, scale) in features.items():
        buckets = weather_buckets[col] if source == "weather" else wofost_buckets[col]
        for k in range(num_buckets):
            out[f"{name}_{prefix}{k:02d}"] = buckets.iloc[k] * scale
    return out


def process_wofost_dataset(
    manifest_path: Optional[str] = None,
    aggregation: Union[str, int] = "weekly",
    align: str = "season_start",
    weather_dir: Optional[str] = None,
    wofost_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> dict:
    """Build one processed table per crop from a batch-run manifest.

    Only `status == "success"` episodes are used. Writes, per crop:
    `data/processed/wofost_{crop}_{aggregation}.csv` and
    `data/processed/wofost_{crop}_{aggregation}_columns.json` (column groups:
    ids/metadata, static features, time-series features per variable,
    targets). Returns `{crop: csv_path}`.
    """
    wofost_dir = wofost_dir if wofost_dir is not None else DEFAULT_WOFOST_SAVE_DIR
    manifest_path = manifest_path if manifest_path is not None else os.path.join(wofost_dir, "dataset_manifest.csv")
    weather_dir = weather_dir if weather_dir is not None else DEFAULT_WEATHER_CACHE_DIR
    output_dir = output_dir if output_dir is not None else DEFAULT_PROCESSED_DIR
    if align not in ("season_start", "sowing"):
        raise ValueError(f"align must be 'season_start' or 'sowing', got {align!r}.")

    bucket_days, prefix = _resolve_aggregation(aggregation)
    aggregation_name = aggregation if isinstance(aggregation, str) and not str(aggregation).isdigit() else f"{bucket_days}d"
    features = default_timeseries_features()

    manifest = pd.read_csv(manifest_path)
    episodes = manifest[manifest["status"] == "success"]
    print(f"{len(episodes)}/{len(manifest)} successful episodes in {manifest_path}.")

    rows, skipped = [], []
    for (longitude, latitude), location_episodes in episodes.groupby(["longitude", "latitude"], sort=False):
        try:
            weather = _load_weather(longitude, latitude, weather_dir)
        except Exception as e:
            skipped += [(r["location_index"], r["year"], f"weather cache: {e}") for _, r in location_episodes.iterrows()]
            continue
        for _, row in location_episodes.iterrows():
            try:
                rows.append(process_episode(row, weather, wofost_dir, bucket_days, prefix, align, features))
            except Exception as e:
                skipped.append((row["location_index"], row["year"], f"{type(e).__name__}: {e}"))

    for location_index, year, error in skipped:
        print(f"  skipped location {location_index}, sowing year {year}: {error}")
    if not rows:
        raise ValueError("No episodes could be processed.")

    os.makedirs(output_dir, exist_ok=True)
    written = {}
    for crop in sorted({r["crop"] for r in rows}):
        # Built per crop (not as one frame split afterwards) so each crop keeps
        # exactly its own bucket columns, even buckets that are all NaN.
        df = pd.DataFrame([r for r in rows if r["crop"] == crop])
        df = df.sort_values(["location_index", "year"]).reset_index(drop=True)
        stem = os.path.join(output_dir, f"wofost_{crop}_{aggregation_name}")
        df.to_csv(f"{stem}.csv", index=False)

        timeseries = {
            name: [c for c in df.columns if c.startswith(f"{name}_{prefix}") and c[len(name) + 1 + len(prefix):].isdigit()]
            for name in features
        }
        num_missing_weather = int(df[[c for n, cols in timeseries.items() if features[n][1] == "weather" for c in cols]].isna().any(axis=1).sum())
        columns = {
            "row": "one simulated growing season (sowing -> maturity) at one location; year = harvest year",
            "aggregation": aggregation_name,
            "bucket_days": bucket_days,
            "num_buckets": len(next(iter(timeseries.values()))),
            "align": align,
            "ids": ["sample_id", "location_index", "location_id", "crop", "year", "sowing_year"],
            "metadata": ["sowing_date", "maturity_date", "season_length_days", "reached_maturity", "window_start"],
            "static_features": default_static_features(),
            "timeseries_features": timeseries,
            "targets": ["yield_t_per_ha", "yield_kg_per_ha"],
        }
        with open(f"{stem}_columns.json", "w") as f:
            json.dump(columns, f, indent=2)

        print(
            f"{crop}: {len(df)} seasons x {columns['num_buckets']} {aggregation_name} buckets -> {stem}.csv"
            + (f" ({num_missing_weather} rows with some missing weather)" if num_missing_weather else "")
        )
        written[crop] = f"{stem}.csv"

    return written


def main():
    if len(sys.argv) == 1:
        print("No arguments provided!")
        print(
            "Usage: python process_wofost_dataset.py [--manifest <path>] "
            "[--aggregation weekly|dekadal|biweekly|monthly|daily|<days>] [--align season_start|sowing] "
            "[--weather-dir <path>] [--wofost-dir <path>] [--output-dir <path>]"
        )
        print("Example: python process_wofost_dataset.py --manifest data/raw/wofost/dataset_manifest.csv --aggregation weekly")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Build per-crop training tables (one row per growing season) from a WOFOST batch run.")
    parser.add_argument("--manifest", dest="manifest_path", type=str, default=None, help="dataset_manifest.csv from generate_wofost_dataset (default: data/raw/wofost/dataset_manifest.csv).")
    parser.add_argument("--aggregation", dest="aggregation", type=str, default="weekly", help="Time-series bucket: weekly (default), dekadal, biweekly, monthly (30 days), daily, or a number of days.")
    parser.add_argument("--align", dest="align", type=str, default="season_start", choices=["season_start", "sowing"], help="Where bucket 0 starts: the crop's nominal season start (default) or the actual sowing date.")
    parser.add_argument("--weather-dir", dest="weather_dir", type=str, default=DEFAULT_WEATHER_CACHE_DIR)
    parser.add_argument("--wofost-dir", dest="wofost_dir", type=str, default=DEFAULT_WOFOST_SAVE_DIR, help="Fallback location of the episode files if the manifest paths have moved.")
    parser.add_argument("-o", "--output-dir", dest="output_dir", type=str, default=DEFAULT_PROCESSED_DIR)

    args = parser.parse_args()
    process_wofost_dataset(
        manifest_path=args.manifest_path,
        aggregation=int(args.aggregation) if args.aggregation.isdigit() else args.aggregation,
        align=args.align,
        weather_dir=args.weather_dir,
        wofost_dir=args.wofost_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
