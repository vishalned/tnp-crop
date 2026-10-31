"""Download the raw SoilGrids values of a locations CSV once and keep them (one table, not one file per location).

The soil files of the pipeline (`generate_gee_soil_file.py`) are derived from these values, and the derivation had a
bug (organic matter = SOC x 0.58 instead of SOC / 0.58). Only the derived YAMLs were kept, so fixing the formula
meant downloading SoilGrids again; this script keeps the raw values so the soil can be rederived offline from now on
(set `$TNP_RAW_SOIL` to the table, see `generate_gee_soil_file._raw_soilgrids`).

    uv run python -m src.data_pipeline.soil.download_raw_soilgrids \
        --locations-csv data/raw/locations/locations_maize_eu27_dense700_s42.csv \
        --out /projects/prjs1788/tnp-crop/data/raw/soilgrids_raw/soilgrids_raw_maize_eu27_dense700_s42.parquet

Output: one row per location (`location_index` = row number of the CSV) and depth layer, with `latitude`,
`longitude`, `zmin`, `zmax` and the 7 SoilGrids variables in conventional units (what `calculate_van_genuchten`
takes). Resumable: a `<out>.partial` file is written every `--checkpoint-every` locations and read back at the start;
locations whose download failed are listed in `<out>.failed.csv`. Points are requested by `--workers` threads.
"""

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from src.data_pipeline.soil.utils_soil.gee_soilgrids import get_df_soilgrids_gee

RETRIES = 3


def fetch(index: int, longitude: float, latitude: float) -> pd.DataFrame:
    for attempt in range(RETRIES):
        try:
            df = get_df_soilgrids_gee(None, longitude=longitude, latitude=latitude)
            df.insert(0, "location_index", index)
            return df
        except Exception:
            if attempt == RETRIES - 1:
                raise
            time.sleep(5 * (attempt + 1))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--locations-csv", required=True)
    p.add_argument("--out", required=True, help="output parquet")
    p.add_argument("--first-location", type=int, default=0)
    p.add_argument("--max-locations", type=int, default=None, help="rows [first, max) of the CSV (default: all)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--checkpoint-every", type=int, default=500)
    args = p.parse_args()

    locations = pd.read_csv(args.locations_csv)
    rows = range(args.first_location, args.max_locations or len(locations))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    partial = args.out + ".partial"
    frames = [pd.read_parquet(partial)] if os.path.exists(partial) else []
    done = set(frames[0]["location_index"]) if frames else set()
    todo = [i for i in rows if i not in done]
    print(f"{len(rows)} locations, {len(done)} already downloaded, {len(todo)} to go", flush=True)

    failed, n = [], 0
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(fetch, i, locations.longitude.iloc[i], locations.latitude.iloc[i]): i for i in todo}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                frames.append(fut.result())
            except Exception as e:
                failed.append({"location_index": i, "error": f"{type(e).__name__}: {e}"[:300]})
                print(f"location {i} FAILED: {failed[-1]['error']}", flush=True)
            n += 1
            if n % args.checkpoint_every == 0:
                pd.concat(frames, ignore_index=True).to_parquet(partial)
                print(f"{n}/{len(todo)} done ({time.strftime('%H:%M:%S')})", flush=True)

    table = pd.concat(frames, ignore_index=True).sort_values(["location_index", "zmin"]).reset_index(drop=True)
    table.to_parquet(args.out)
    pd.DataFrame(failed, columns=["location_index", "error"]).to_csv(args.out + ".failed.csv", index=False)
    if os.path.exists(partial):
        os.remove(partial)
    print(f"{table['location_index'].nunique()} locations ({len(table)} rows) -> {args.out}; {len(failed)} failed", flush=True)


if __name__ == "__main__":
    main()
