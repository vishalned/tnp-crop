"""Turn a WOFOST batch run (its `dataset_manifest.csv` plus the per-episode
daily CSVs and per-location weather caches under data/raw) into one
training table per crop under data/processed.

One row = one simulated growing season (sowing -> maturity) at one
location, the same unit as a CYBench row (region x season). Rows are
labelled with the harvest year (the year the crop matured), CYBench's
convention: winter wheat sown in Oct 2013 and harvested in Jul 2014 is
year 2014. The sowing year is kept as `sowing_year`.

Time series are stored at daily resolution, unaggregated -- any
aggregation (weekly, dekadal, ...) is left to the dataloader. Every row of
a crop has the same number of days, anchored on the season start:

- `align="season_start"` (default): anchored on the crop's nominal start of
  season for that year (`default_sowing_doy()`, the stand-in for the
  WorldCereal SOS that CYBench also aligns to). The actual, jittered
  sowing date then shows up inside the series, as it would in real data.
- `align="sowing"`: anchored on the actual sowing date.

Day 0 is `pre_season_days` (default 30) before that anchor, so a sowing
that the jitter moved earlier than the nominal start is still fully inside
the window, and the model also sees some pre-season weather. The window
then runs `default_max_duration_days()` past the anchor (the longest a
season can run), so it doesn't depend on when the crop matured. Weather for the whole
window comes from the per-location weather cache, so it is complete even
after maturity. Simulated crop/soil variables only exist while the crop is
growing: `fpar` is 0 outside it (bare field before sowing / after harvest),
`ssm` is left NaN there (it isn't simulated).

Time-series column names are `{feature}_d{day:03d}`, e.g. `tmax_d000` for
the first day. The exact columns of each group are written to a
`..._columns.json` sidecar next to the table, for the dataloader.
"""

import argparse
import datetime
import json
import os
import sys
from typing import Optional

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


def default_timeseries_features() -> dict:
    """CYBench-aligned daily time-series features: output name -> (source
    column, source, scale factor to the unit given here).

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


def process_episode(
    row: pd.Series,
    weather: pd.DataFrame,
    wofost_dir: str,
    align: str,
    pre_season_days: int,
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
    anchor = season_start if align == "season_start" else sowing_date
    window_start = anchor - datetime.timedelta(days=pre_season_days)
    num_days = pre_season_days + default_max_duration_days()[crop]
    window_days = pd.date_range(window_start, periods=num_days, freq="D").date

    matured = daily.index[daily["DVS"] >= 2.0]
    reached_maturity = len(matured) > 0
    maturity_date = matured[0] if reached_maturity else daily.index[-1]

    weather_window = weather.reindex(window_days)
    wofost_window = daily.reindex(window_days)
    if "fpar" in wofost_window:
        # No green canopy outside the simulated crop life: before sowing and
        # after maturity (harvest) the field is bare.
        wofost_window["fpar"] = wofost_window["fpar"].fillna(0.0)

    # Manifests from before per-year jitters had one episode per year.
    jitter_index = int(row["jitter_index"]) if pd.notna(row.get("jitter_index")) else 0
    out = {
        "sample_id": f"{row['longitude']}_{row['latitude']}_{crop}_{maturity_date.year}_j{jitter_index}",
        "location_index": int(row["location_index"]),
        "location_id": f"{row['longitude']}_{row['latitude']}",
        "crop": crop,
        "year": maturity_date.year,
        "sowing_year": sowing_year,
        "jitter_index": jitter_index,
        "latitude": row["latitude"],
        "longitude": row["longitude"],
        "awc": row["awc"],
        "bulk_density": row["bulk_density"],
        "sos_doy": sos_doy,
        "sowing_doy": sowing_date.timetuple().tm_yday,
        "sowing_date": sowing_date.isoformat(),
        "sowing_offset_days": (sowing_date - season_start).days,
        "maturity_date": maturity_date.isoformat(),
        "season_length_days": (maturity_date - sowing_date).days,
        "reached_maturity": reached_maturity,
        "window_start": window_start.isoformat(),
        "yield_kg_per_ha": row["yield_kg_per_ha"],
        "yield_t_per_ha": row["yield_kg_per_ha"] / 1000.0,
    }
    for name, (col, source, scale) in features.items():
        values = (weather_window if source == "weather" else wofost_window)[col].to_numpy() * scale
        out.update({f"{name}_d{k:03d}": v for k, v in enumerate(values)})
    return out


def process_wofost_dataset(
    manifest_path: Optional[str] = None,
    align: str = "season_start",
    pre_season_days: int = 30,
    weather_dir: Optional[str] = None,
    wofost_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> dict:
    """Build one processed table per crop from a batch-run manifest.

    Only `status == "success"` episodes are used. Writes, per crop:
    `data/processed/wofost_{crop}_daily.csv` and
    `data/processed/wofost_{crop}_daily_columns.json` (column groups:
    ids/metadata, static features, time-series features per variable,
    targets). Returns `{crop: csv_path}`.
    """
    wofost_dir = wofost_dir if wofost_dir is not None else DEFAULT_WOFOST_SAVE_DIR
    manifest_path = manifest_path if manifest_path is not None else os.path.join(wofost_dir, "dataset_manifest.csv")
    weather_dir = weather_dir if weather_dir is not None else DEFAULT_WEATHER_CACHE_DIR
    output_dir = output_dir if output_dir is not None else DEFAULT_PROCESSED_DIR
    if align not in ("season_start", "sowing"):
        raise ValueError(f"align must be 'season_start' or 'sowing', got {align!r}.")
    if pre_season_days < 0:
        raise ValueError(f"pre_season_days must be >= 0, got {pre_season_days}.")

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
                rows.append(process_episode(row, weather, wofost_dir, align, pre_season_days, features))
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
        # exactly its own day columns, even days that are all NaN.
        df = pd.DataFrame([r for r in rows if r["crop"] == crop])
        df = df.sort_values(["location_index", "year", "jitter_index"]).reset_index(drop=True)
        stem = os.path.join(output_dir, f"wofost_{crop}_daily")
        df.to_csv(f"{stem}.csv", index=False)

        timeseries = {
            name: [c for c in df.columns if c.startswith(f"{name}_d") and c[len(name) + 2:].isdigit()]
            for name in features
        }
        num_missing_weather = int(df[[c for n, cols in timeseries.items() if features[n][1] == "weather" for c in cols]].isna().any(axis=1).sum())
        columns = {
            "row": "one simulated growing season (sowing -> maturity) at one location; year = harvest year",
            "resolution": "daily (unaggregated)",
            "num_days": len(next(iter(timeseries.values()))),
            "align": align,
            "pre_season_days": pre_season_days,
            "anchor_day_index": pre_season_days,
            "ids": ["sample_id", "location_index", "location_id", "crop", "year", "sowing_year", "jitter_index"],
            "metadata": ["sowing_date", "sowing_offset_days", "maturity_date", "season_length_days", "reached_maturity", "window_start"],
            "static_features": default_static_features(),
            "timeseries_features": timeseries,
            "targets": ["yield_t_per_ha", "yield_kg_per_ha"],
        }
        with open(f"{stem}_columns.json", "w") as f:
            json.dump(columns, f, indent=2)

        print(
            f"{crop}: {len(df)} seasons x {columns['num_days']} days -> {stem}.csv"
            + (f" ({num_missing_weather} rows with some missing weather)" if num_missing_weather else "")
        )
        written[crop] = f"{stem}.csv"

    return written


def main():
    if len(sys.argv) == 1:
        print("No arguments provided!")
        print(
            "Usage: python process_wofost_dataset.py [--manifest <path>] "
            "[--align season_start|sowing] [--pre-season-days <n>] "
            "[--weather-dir <path>] [--wofost-dir <path>] [--output-dir <path>]"
        )
        print("Example: python process_wofost_dataset.py --manifest data/raw/wofost/dataset_manifest.csv")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Build per-crop training tables (one row per growing season) from a WOFOST batch run.")
    parser.add_argument("--manifest", dest="manifest_path", type=str, default=None, help="dataset_manifest.csv from generate_wofost_dataset (default: data/raw/wofost/dataset_manifest.csv).")
    parser.add_argument("--align", dest="align", type=str, default="season_start", choices=["season_start", "sowing"], help="What the window is anchored on: the crop's nominal season start (default) or the actual sowing date.")
    parser.add_argument("--pre-season-days", dest="pre_season_days", type=int, default=30, help="Days before the anchor that day 0 starts (default 30; covers sowing-date jitter and some pre-season weather).")
    parser.add_argument("--weather-dir", dest="weather_dir", type=str, default=DEFAULT_WEATHER_CACHE_DIR)
    parser.add_argument("--wofost-dir", dest="wofost_dir", type=str, default=DEFAULT_WOFOST_SAVE_DIR, help="Fallback location of the episode files if the manifest paths have moved.")
    parser.add_argument("-o", "--output-dir", dest="output_dir", type=str, default=DEFAULT_PROCESSED_DIR)

    args = parser.parse_args()
    process_wofost_dataset(
        manifest_path=args.manifest_path,
        align=args.align,
        pre_season_days=args.pre_season_days,
        weather_dir=args.weather_dir,
        wofost_dir=args.wofost_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
