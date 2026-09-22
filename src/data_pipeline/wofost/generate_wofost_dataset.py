"""Batch runner: generate WOFOST episodes for many locations x years and
write them all under data/raw/wofost with a single manifest indexing every
attempted episode -- the "sanity-check-then-scale" step in
wofost_synthetic_pretraining_plan, run at whatever scale you point it at.
"""

import argparse
import os
import random
import sys
import traceback
from typing import Optional

import pandas as pd
import rootutils

from src.data_pipeline.wofost.run_wofost_simulation import DEFAULT_WOFOST_SAVE_DIR, generate_wofost_episode


root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def _sample_years(rng: random.Random, start_year: int, end_year: int, num_years: int) -> list:
    available = end_year - start_year + 1
    k = min(num_years, available)
    return sorted(rng.sample(range(start_year, end_year + 1), k=k))


def generate_wofost_dataset(
    locations: list,
    crop: Optional[str] = None,
    num_years: int = 5,
    start_year: int = 2000,
    end_year: int = 2023,
    sowing_jitter_days: int = 10,
    seed: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> dict:
    """Run WOFOST episodes for a batch of locations, `num_years` years
    sampled (without replacement) per location from `[start_year, end_year]`,
    and write one manifest CSV (`dataset_manifest.csv`, one row per attempted
    episode) alongside the per-episode files `generate_wofost_episode`
    already writes.

    :param locations: list of dicts, each with `longitude`/`latitude` and
        optionally `crop` (overrides `crop` for that location).
    :param crop: default crop for locations that don't specify their own.

    A single location/year failure (e.g. a GEE quota error) doesn't stop the
    batch -- it's recorded in the manifest with `status="failed"` and the
    run moves on, since one bad episode shouldn't lose a whole batch.
    """
    save_dir = output_dir if output_dir is not None else DEFAULT_WOFOST_SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)
    manifest_path = os.path.join(save_dir, "dataset_manifest.csv")

    rng = random.Random(seed)
    manifest_rows = []

    for i, location in enumerate(locations):
        longitude = location["longitude"]
        latitude = location["latitude"]
        raw_crop = location.get("crop")
        location_crop = raw_crop if pd.notna(raw_crop) else crop
        if not location_crop:
            raise ValueError(
                f"No crop given for location #{i} ({longitude}, {latitude}) and no default --crop set."
            )

        for year in _sample_years(rng, start_year, end_year, num_years):
            episode_seed = rng.randint(0, 2**31 - 1)
            print(f"[{i + 1}/{len(locations)}] {location_crop} at ({longitude}, {latitude}), year {year}")

            row = {
                "longitude": longitude,
                "latitude": latitude,
                "crop": location_crop,
                "year": year,
                "seed": episode_seed,
                "status": "failed",
                "error": None,
                "sowing_date": None,
                "yield_kg_per_ha": None,
                "final_dvs": None,
                "awc": None,
                "bulk_density": None,
                "daily_path": None,
                "summary_path": None,
            }
            try:
                result = generate_wofost_episode(
                    longitude=longitude,
                    latitude=latitude,
                    crop=location_crop,
                    year=year,
                    sowing_jitter_days=sowing_jitter_days,
                    seed=episode_seed,
                    output_dir=save_dir,
                )
                summary = result["summary"]
                row.update(
                    status="success",
                    sowing_date=summary.get("sowing_date"),
                    yield_kg_per_ha=summary.get("yield_kg_per_ha"),
                    final_dvs=summary.get("final_dvs"),
                    awc=summary.get("awc"),
                    bulk_density=summary.get("bulk_density"),
                    daily_path=result["daily_path"],
                    summary_path=result["summary_path"],
                )
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"
                print(f"  FAILED: {row['error']}")
                traceback.print_exc()

            manifest_rows.append(row)
            # Write after every episode so a crash partway through a large
            # batch doesn't lose the progress already made.
            pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    num_success = sum(1 for r in manifest_rows if r["status"] == "success")
    print(f"Done: {num_success}/{len(manifest_rows)} episodes succeeded. Manifest written to {manifest_path}.")

    return {"manifest_path": manifest_path, "rows": manifest_rows}


def _load_locations_csv(path: str) -> list:
    df = pd.read_csv(path)
    if "longitude" not in df.columns or "latitude" not in df.columns:
        raise ValueError(f"Locations CSV {path} must have 'longitude' and 'latitude' columns.")
    return df.to_dict(orient="records")


def main():
    if len(sys.argv) == 1:
        print("No arguments provided!")
        print(
            "Usage: python generate_wofost_dataset.py --locations-csv <path> [--crop <wheat|maize>] "
            "[--num-years <n>] [--start-year <y>] [--end-year <y>] [--sowing-jitter-days <days>] "
            "[--seed <int>] [--output-dir <path>]"
        )
        print(
            "The locations CSV needs 'longitude'/'latitude' columns, and an optional 'crop' "
            "column to override --crop per row."
        )
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Run WOFOST episodes for a batch of locations x years and write a dataset manifest."
    )
    parser.add_argument("--locations-csv", dest="locations_csv", type=str, required=True, help="CSV with 'longitude'/'latitude' columns (and optional 'crop' column).")
    parser.add_argument("--crop", dest="crop", type=str, default=None, choices=["wheat", "maize"], help="Default crop for locations that don't specify their own.")
    parser.add_argument("--num-years", dest="num_years", type=int, default=5, help="Number of years sampled per location.")
    parser.add_argument("--start-year", dest="start_year", type=int, default=2000)
    parser.add_argument("--end-year", dest="end_year", type=int, default=2023)
    parser.add_argument("--sowing-jitter-days", dest="sowing_jitter_days", type=int, default=10)
    parser.add_argument("--seed", dest="seed", type=int, default=None)
    parser.add_argument("-o", "--output-dir", dest="output_dir", type=str, default=DEFAULT_WOFOST_SAVE_DIR)

    args = parser.parse_args()

    locations = _load_locations_csv(args.locations_csv)

    generate_wofost_dataset(
        locations=locations,
        crop=args.crop,
        num_years=args.num_years,
        start_year=args.start_year,
        end_year=args.end_year,
        sowing_jitter_days=args.sowing_jitter_days,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
