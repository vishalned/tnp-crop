"""Batch runner: generate WOFOST episodes for many locations x years x
sowing-date jitters and write them all under data/raw/wofost with a single
manifest indexing every attempted episode.

Every location gets every year in `[start_year, end_year]` (2005-2020 by
default) and, per year, `num_jitters` (3) simulations with distinct sowing
dates within +/- `sowing_jitter_days` of the crop's season start -- so
`locations x 16 years x 3 jitters` episodes. Sub-sampling years/jitters is
left to the dataloader.
"""

import argparse
import datetime
import multiprocessing
import os
import random
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import pandas as pd
import rootutils

from src.data_pipeline.weather.utils_weather.gee_weather import get_gee_weather_provider_for_location
from src.data_pipeline.wofost.run_wofost_simulation import DEFAULT_WOFOST_SAVE_DIR, generate_wofost_episode
from src.data_pipeline.wofost.utils_wofost.default_wofost_variables import (
    default_max_duration_days,
    default_sowing_doy,
)


root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def _sample_jitter_offsets(rng: random.Random, num_jitters: int, sowing_jitter_days: int) -> list:
    """`num_jitters` distinct sowing offsets (days) in
    [-sowing_jitter_days, +sowing_jitter_days], sorted. Distinct so no two
    jitters of the same location/year are the same simulation."""
    choices = range(-sowing_jitter_days, sowing_jitter_days + 1)
    if num_jitters > len(choices):
        raise ValueError(
            f"num_jitters ({num_jitters}) can't exceed the {len(choices)} distinct sowing days "
            f"in +/-{sowing_jitter_days} days."
        )
    return sorted(rng.sample(choices, k=num_jitters))


def _prefetch_weather_for_window(
    longitude: float, latitude: float, crop: str, years: list, sowing_jitter_days: int
) -> None:
    """Download a location's weather for its whole year window in one go,
    before its episodes run, so each episode is then a cache hit.

    Covers Jan 1 of the first year up to the latest date any episode in the
    window can ask for: the last year's sowing day (+ jitter) + the crop's
    max season length -- e.g. into the following year for winter wheat.
    Failures are only printed: each episode retries the fetch itself and
    records its own failure in the manifest.
    """
    last_needed = datetime.date(years[-1], 1, 1) + datetime.timedelta(
        days=default_sowing_doy()[crop] - 1 + sowing_jitter_days + default_max_duration_days()[crop]
    )
    try:
        get_gee_weather_provider_for_location(
            latitude=latitude,
            longitude=longitude,
            start_date=datetime.date(years[0], 1, 1),
            end_date=last_needed,
        )
    except Exception as e:
        print(f"  Weather prefetch failed ({type(e).__name__}: {e}); episodes will retry individually.")


def _empty_row(job: dict, year: int, jitter_index: int, sowing_offset_days: int) -> dict:
    return {
        "location_index": job["location_index"],
        "longitude": job["longitude"],
        "latitude": job["latitude"],
        "crop": job["crop"],
        "year": year,
        "jitter_index": jitter_index,
        "sowing_offset_days": sowing_offset_days,
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


def _run_location_group(jobs: list, sowing_jitter_days: int, save_dir: str, num_locations: int) -> list:
    """Run every episode for a group of location jobs that share the same
    coordinates (e.g. one point listed once for wheat and once for maize),
    one after the other, and return their manifest rows.

    Grouping by coordinates is what makes the parallel run safe: the soil
    and weather caches are one file per (longitude, latitude), so two
    workers never write the same cache file at the same time.

    Module-level (not a closure) so it can be sent to worker processes.
    """
    rows = []
    for job in jobs:
        longitude, latitude, crop, years = job["longitude"], job["latitude"], job["crop"], job["years"]
        tag = f"[{job['location_index'] + 1}/{num_locations}] {crop} at ({longitude}, {latitude})"
        print(f"{tag}, years {years[0]}-{years[-1]}", flush=True)
        _prefetch_weather_for_window(longitude, latitude, crop, years, sowing_jitter_days)

        for year, jitter_index, offset in _episodes(job):
            print(f"{tag}, year {year}, jitter {jitter_index} ({offset:+d} days)", flush=True)
            row = _empty_row(job, year, jitter_index, offset)
            try:
                result = generate_wofost_episode(
                    longitude=longitude,
                    latitude=latitude,
                    crop=crop,
                    year=year,
                    sowing_offset_days=offset,
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
                print(f"{tag}, year {year}, jitter {jitter_index} FAILED: {row['error']}", flush=True)
                traceback.print_exc()
            rows.append(row)
    return rows


def _episodes(job: dict):
    """(year, jitter_index, sowing_offset_days) for every episode of a job."""
    for year, offsets in zip(job["years"], job["sowing_offsets"]):
        for jitter_index, offset in enumerate(offsets):
            yield year, jitter_index, offset


def _default_num_workers() -> int:
    return min(8, os.cpu_count() or 1)


def generate_wofost_dataset(
    locations: list,
    crop: Optional[str] = None,
    start_year: int = 2005,
    end_year: int = 2020,
    num_jitters: int = 3,
    sowing_jitter_days: int = 10,
    seed: Optional[int] = None,
    output_dir: Optional[str] = None,
    num_workers: Optional[int] = None,
) -> dict:
    """Run WOFOST episodes for a batch of locations -- every year in
    `[start_year, end_year]` (sowing year), `num_jitters` sowing dates per
    year -- and write one manifest CSV (`dataset_manifest.csv`, one row per
    attempted episode, with `jitter_index` and `sowing_offset_days`)
    alongside the per-episode files `generate_wofost_episode` already writes.

    :param locations: list of dicts, each with `longitude`/`latitude` and
        optionally `crop` (overrides `crop` for that location).
    :param crop: default crop for locations that don't specify their own.
    :param num_jitters: sowing dates simulated per location and year: distinct
        random offsets within +/- `sowing_jitter_days` of the crop's season
        start (`jitter_index` 0..num_jitters-1, ordered by offset).
    :param num_workers: number of worker processes; locations are spread
        over them (all rows with the same coordinates go to the same
        worker). 1 runs everything in this process, one location after the
        other. Defaults to `min(8, cpu count)`.

    Sowing offsets are all drawn up front in this process, in location
    order, so a given `seed` gives the same episodes whatever `num_workers`
    is.

    A single location/year failure (e.g. a GEE quota error) doesn't stop the
    batch -- it's recorded in the manifest with `status="failed"` and the
    run moves on, since one bad episode shouldn't lose a whole batch.
    """
    save_dir = output_dir if output_dir is not None else DEFAULT_WOFOST_SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)
    manifest_path = os.path.join(save_dir, "dataset_manifest.csv")
    num_workers = num_workers if num_workers is not None else _default_num_workers()
    if end_year < start_year:
        raise ValueError(f"end_year ({end_year}) must be >= start_year ({start_year}).")
    years = list(range(start_year, end_year + 1))

    rng = random.Random(seed)
    groups = {}
    for i, location in enumerate(locations):
        longitude = location["longitude"]
        latitude = location["latitude"]
        raw_crop = location.get("crop")
        location_crop = raw_crop if pd.notna(raw_crop) else crop
        if not location_crop:
            raise ValueError(
                f"No crop given for location #{i} ({longitude}, {latitude}) and no default --crop set."
            )
        job = {
            "location_index": i,
            "longitude": longitude,
            "latitude": latitude,
            "crop": location_crop,
            "years": years,
            "sowing_offsets": [_sample_jitter_offsets(rng, num_jitters, sowing_jitter_days) for _ in years],
        }
        groups.setdefault((longitude, latitude), []).append(job)

    manifest_rows = []

    def record(rows: list) -> None:
        manifest_rows.extend(rows)
        # Written after every finished location so a crash partway through
        # a large batch doesn't lose the progress already made.
        (
            pd.DataFrame(manifest_rows)
            .sort_values(["location_index", "year", "jitter_index"])
            .to_csv(manifest_path, index=False)
        )

    group_list = list(groups.values())
    if num_workers <= 1:
        for jobs in group_list:
            record(_run_location_group(jobs, sowing_jitter_days, save_dir, len(locations)))
    else:
        print(f"Running {len(group_list)} locations on {num_workers} worker processes.", flush=True)
        # spawn (not fork): each worker starts clean and initialises its own
        # Earth Engine client, rather than inheriting a forked copy.
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=context) as executor:
            futures = {
                executor.submit(_run_location_group, jobs, sowing_jitter_days, save_dir, len(locations)): jobs
                for jobs in group_list
            }
            for future in as_completed(futures):
                try:
                    rows = future.result()
                except Exception as e:
                    # The worker process itself died (e.g. out of memory);
                    # mark every episode it was responsible for as failed.
                    error = f"worker crashed: {type(e).__name__}: {e}"
                    print(error, flush=True)
                    rows = []
                    for job in futures[future]:
                        for year, jitter_index, offset in _episodes(job):
                            row = _empty_row(job, year, jitter_index, offset)
                            row["error"] = error
                            rows.append(row)
                record(rows)

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
            "[--start-year <y>] [--end-year <y>] [--num-jitters <n>] [--sowing-jitter-days <days>] "
            "[--seed <int>] [--workers <n>] [--output-dir <path>]"
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
    parser.add_argument("--start-year", dest="start_year", type=int, default=2005, help="First sowing year (default 2005).")
    parser.add_argument("--end-year", dest="end_year", type=int, default=2020, help="Last sowing year (default 2020).")
    parser.add_argument("--num-jitters", dest="num_jitters", type=int, default=3, help="Distinct sowing dates simulated per location and year (default 3).")
    parser.add_argument("--sowing-jitter-days", dest="sowing_jitter_days", type=int, default=10, help="Sowing dates are drawn within +/- this many days of the season start (default 10).")
    parser.add_argument("--seed", dest="seed", type=int, default=None)
    parser.add_argument("--workers", dest="num_workers", type=int, default=None, help="Worker processes running locations in parallel (default: min(8, cpu count); 1 = sequential).")
    parser.add_argument("-o", "--output-dir", dest="output_dir", type=str, default=DEFAULT_WOFOST_SAVE_DIR)

    args = parser.parse_args()

    locations = _load_locations_csv(args.locations_csv)

    generate_wofost_dataset(
        locations=locations,
        crop=args.crop,
        start_year=args.start_year,
        end_year=args.end_year,
        num_jitters=args.num_jitters,
        sowing_jitter_days=args.sowing_jitter_days,
        seed=args.seed,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
