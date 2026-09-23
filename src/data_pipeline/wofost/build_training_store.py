"""Build the compact training store the TNP dataloader samples episodes from.

Inputs (all produced earlier in the pipeline):
- the batch run's `dataset_manifest.csv` (one row per location x sowing
  year x jitter),
- the locations CSV the batch ran on (`extract_locations_from_zarr.py`
  output: gives each location its `country` and `zarr_index`),
- the per-location GEE weather caches (`data/raw/weather`),
- the CropFM dataset Zarr (static soil + terrain per `zarr_index`),
- the episodes' daily WOFOST CSVs (for the flowering/maturity dates).

Output, `data/processed/tnp_store_{crop}/`:
- `points.csv`: one row per location: ids, `country`, coordinates and the
  static variables (`clay_0..2`, `nitrogen_0..2`, `ph_0..2`, `soc_0..2` for
  the Zarr's 0-5/5-15/15-30 cm layers, `water_holding_capacity`,
  `elevation`, `slope`).
- `seasons.csv`: one row per successful episode (point x season year x
  jitter): sowing date, nominal season start, flowering/maturity dates and
  days after the season start, yield (t/ha), harvest year.
- `weather.npy`: float32 `[num_points, num_days, 6]` daily weather on one
  shared date axis (`weather_meta.json`): tmin/tmax (degC), precip
  (mm/day), radiation (MJ/m2/day), wind (m/s, 2 m), humidity (vapour
  pressure, hPa). Stored once per location, since every jitter of a
  location/year runs on the same weather.

`season_year` is the sowing year (the year the manifest and the year
split use); `harvest_year` is the year the crop matured.
"""

import argparse
import datetime
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import numpy as np
import pandas as pd
import rootutils

from src.data.components.crop_vocab import WEATHER_VARIABLES
from src.data_pipeline.weather.utils_weather.gee_weather import DEFAULT_WEATHER_CACHE_DIR, _weather_cache_path
from src.data_pipeline.wofost.run_wofost_simulation import DEFAULT_WOFOST_SAVE_DIR


root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
DEFAULT_ZARR_PATH = "/projects/prjs1788/CropFM/europe_data_1200_merged_cleaned2_rechunked.zarr"


def default_store_dir(crop: str) -> str:
    return os.path.join(str(root), "data", "processed", f"tnp_store_{crop}")


def weather_cache_columns() -> dict:
    """Store weather variable -> (weather cache column, scale to store unit)."""
    return {
        "tmin": ("TMIN", 1.0),
        "tmax": ("TMAX", 1.0),
        "precip": ("RAIN", 10.0),  # cm/day -> mm/day
        "radiation": ("IRRAD", 1e-6),  # J/m2/day -> MJ/m2/day
        "wind": ("WIND", 1.0),
        "humidity": ("VAP", 1.0),
    }


def _phenology_dates(daily_path: str) -> tuple:
    """(flowering date, maturity date) = first day DVS reaches 1 / 2, or None."""
    daily = pd.read_csv(daily_path, usecols=["day", "DVS"])
    days = pd.to_datetime(daily["day"]).dt.date.to_numpy()
    dvs = daily["DVS"].to_numpy()
    flowering = days[np.argmax(dvs >= 1.0)] if (dvs >= 1.0).any() else None
    maturity = days[np.argmax(dvs >= 2.0)] if (dvs >= 2.0).any() else None
    return flowering, maturity


def _resolve_path(path: str, fallback_dir: str) -> str:
    if isinstance(path, str) and os.path.exists(path):
        return path
    candidate = os.path.join(fallback_dir, os.path.basename(str(path)))
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(f"Episode file not found: {path} (also tried {candidate}).")


def _load_static_from_zarr(zarr_path: str, zarr_index: np.ndarray) -> pd.DataFrame:
    import zarr

    store = zarr.open(zarr_path, mode="r")
    out = {}
    for name, key in [("clay", "clay"), ("nitrogen", "nitrogen"), ("ph", "phh2o"), ("soc", "soc")]:
        values = np.asarray(store[f"static_modalities/soil/{key}"][:], dtype=np.float32)[zarr_index]
        values = values.reshape(len(zarr_index), -1)
        for layer in range(values.shape[1]):
            out[f"{name}_{layer}"] = values[:, layer]
    for name in ["elevation", "slope"]:
        out[name] = np.asarray(store[f"static_modalities/elevation/{name}"][:], dtype=np.float32)[zarr_index]
    return pd.DataFrame(out)


def build_training_store(
    manifest_path: str,
    locations_csv: str,
    crop: Optional[str] = None,
    zarr_path: str = DEFAULT_ZARR_PATH,
    weather_dir: Optional[str] = None,
    wofost_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    num_workers: int = 1,
) -> str:
    weather_dir = weather_dir or DEFAULT_WEATHER_CACHE_DIR
    wofost_dir = wofost_dir or DEFAULT_WOFOST_SAVE_DIR

    manifest = pd.read_csv(manifest_path)
    if crop is None:
        crops = manifest["crop"].dropna().unique()
        if len(crops) != 1:
            raise ValueError(f"Manifest has crops {list(crops)}; pass --crop to pick one.")
        crop = crops[0]
    episodes = manifest[(manifest["crop"] == crop) & (manifest["status"] == "success")].copy()
    if "jitter_index" not in episodes:
        episodes["jitter_index"] = 0
    print(f"{len(episodes)} successful {crop} episodes in {manifest_path}.")

    # --- points: location metadata + static variables ---------------------
    # The batch runner's location_index is the row number in its locations CSV.
    locations = pd.read_csv(locations_csv)
    for col in ["country", "zarr_index"]:
        if col not in locations:
            raise ValueError(f"{locations_csv} has no '{col}' column (use extract_locations_from_zarr.py output).")
    used = np.sort(episodes["location_index"].unique())
    points = locations.iloc[used][["country", "zarr_index", "longitude", "latitude"]].reset_index(drop=True)
    points.insert(0, "location_index", used)
    coords = episodes.groupby("location_index")[["longitude", "latitude"]].first().loc[used]
    if not np.allclose(coords.to_numpy(), points[["longitude", "latitude"]].to_numpy()):
        raise ValueError("Manifest coordinates don't match the locations CSV rows; is it the CSV the batch ran on?")
    points.insert(0, "point_id", np.arange(len(points)))
    points["crop"] = crop

    static = _load_static_from_zarr(zarr_path, points["zarr_index"].to_numpy())
    awc = episodes.groupby("location_index")["awc"].first().loc[used].to_numpy()
    points = pd.concat([points, static], axis=1)
    points["water_holding_capacity"] = awc
    point_of_location = dict(zip(points["location_index"], points["point_id"]))

    # --- seasons: labels per episode ---------------------------------------
    episodes["point_id"] = episodes["location_index"].map(point_of_location)
    episodes["sowing_date"] = pd.to_datetime(episodes["sowing_date"]).dt.date
    offset = episodes["sowing_offset_days"].fillna(0) if "sowing_offset_days" in episodes else 0
    episodes["season_start"] = [
        s - datetime.timedelta(days=int(o)) for s, o in zip(episodes["sowing_date"], np.broadcast_to(offset, len(episodes)))
    ]
    paths = [_resolve_path(p, os.path.join(wofost_dir, crop)) for p in episodes["daily_path"]]
    print(f"Reading phenology from {len(paths)} daily WOFOST files...")
    if num_workers > 1:
        with ProcessPoolExecutor(num_workers) as ex:
            dates = list(ex.map(_phenology_dates, paths, chunksize=64))
    else:
        dates = [_phenology_dates(p) for p in paths]
    flowering, maturity = zip(*dates)

    seasons = pd.DataFrame({
        "point_id": episodes["point_id"].to_numpy(),
        "season_year": episodes["year"].astype(int).to_numpy(),
        "jitter_index": episodes["jitter_index"].astype(int).to_numpy(),
        "sowing_date": episodes["sowing_date"].to_numpy(),
        "season_start": episodes["season_start"].to_numpy(),
        "flowering_date": flowering,
        "maturity_date": maturity,
        "yield_t_per_ha": episodes["yield_kg_per_ha"].to_numpy() / 1000.0,
    })
    seasons["reached_maturity"] = seasons["maturity_date"].notna()
    for event in ["flowering", "maturity"]:
        seasons[f"{event}_days"] = [
            (d - s).days if d is not None else np.nan for d, s in zip(seasons[f"{event}_date"], seasons["season_start"])
        ]
    seasons["harvest_year"] = [
        d.year if d is not None else y for d, y in zip(seasons["maturity_date"], seasons["season_year"])
    ]
    seasons = seasons.sort_values(["point_id", "season_year", "jitter_index"]).reset_index(drop=True)

    # --- weather: one daily array per point on a shared date axis ----------
    start = min(seasons["season_start"]) - datetime.timedelta(days=120)
    end = max(seasons["season_start"]) + datetime.timedelta(days=430)
    axis = pd.date_range(start, end, freq="D").date
    weather = np.full((len(points), len(axis), len(WEATHER_VARIABLES)), np.nan, dtype=np.float32)
    columns = weather_cache_columns()
    for i, row in points.iterrows():
        cache = pd.read_csv(_weather_cache_path(row["longitude"], row["latitude"], weather_dir))
        cache["DAY"] = pd.to_datetime(cache["DAY"]).dt.date
        cache = cache.set_index("DAY").reindex(axis)
        for v, name in enumerate(WEATHER_VARIABLES):
            col, scale = columns[name]
            weather[i, :, v] = cache[col].to_numpy(dtype=np.float32) * scale

    output_dir = output_dir or default_store_dir(crop)
    os.makedirs(output_dir, exist_ok=True)
    points.to_csv(os.path.join(output_dir, "points.csv"), index=False)
    seasons.to_csv(os.path.join(output_dir, "seasons.csv"), index=False)
    np.save(os.path.join(output_dir, "weather.npy"), weather)
    with open(os.path.join(output_dir, "weather_meta.json"), "w") as f:
        json.dump({
            "start_date": start.isoformat(),
            "num_days": len(axis),
            "variables": WEATHER_VARIABLES,
            "units": {"tmin": "degC", "tmax": "degC", "precip": "mm/day", "radiation": "MJ/m2/day",
                      "wind": "m/s", "humidity": "hPa (vapour pressure)"},
            "crop": crop,
        }, f, indent=2)

    missing = int(np.isnan(weather).any(axis=(1, 2)).sum())
    print(
        f"Store written to {output_dir}: {len(points)} points "
        f"({points['country'].value_counts().to_dict()}), {len(seasons)} seasons, "
        f"weather {weather.shape}" + (f"; {missing} points have some missing weather days" if missing else "")
    )
    return output_dir


def main():
    if len(sys.argv) == 1:
        print("No arguments provided!")
        print(
            "Usage: python build_training_store.py --manifest <dataset_manifest.csv> --locations-csv <locations.csv> "
            "[--crop <wheat|maize>] [--zarr-path <path>] [--weather-dir <path>] [--wofost-dir <path>] "
            "[-o <output dir>] [--workers <n>]"
        )
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Build the TNP training store from a WOFOST batch run.")
    parser.add_argument("--manifest", dest="manifest_path", type=str, required=True)
    parser.add_argument("--locations-csv", dest="locations_csv", type=str, required=True, help="The locations CSV the batch ran on.")
    parser.add_argument("--crop", dest="crop", type=str, default=None, choices=["wheat", "maize"])
    parser.add_argument("--zarr-path", dest="zarr_path", type=str, default=DEFAULT_ZARR_PATH)
    parser.add_argument("--weather-dir", dest="weather_dir", type=str, default=None)
    parser.add_argument("--wofost-dir", dest="wofost_dir", type=str, default=None)
    parser.add_argument("-o", "--output-dir", dest="output_dir", type=str, default=None, help="Default: data/processed/tnp_store_{crop}")
    parser.add_argument("--workers", dest="num_workers", type=int, default=1, help="Processes reading the daily WOFOST files.")
    args = parser.parse_args()

    build_training_store(
        manifest_path=args.manifest_path,
        locations_csv=args.locations_csv,
        crop=args.crop,
        zarr_path=args.zarr_path,
        weather_dir=args.weather_dir,
        wofost_dir=args.wofost_dir,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
