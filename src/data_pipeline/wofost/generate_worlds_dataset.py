"""Generate a WOFOST dataset of sampled worlds (see `worlds.py`) from a config.

    uv run python -m src.data_pipeline.wofost.generate_worlds_dataset --config configs/wofost_worlds/dataset_C.yaml \
        [--location-indices 0 5 9] [--worlds-per-location 2] [--workers 8] [-o <dir>]

Datasets (`configs/wofost_worlds/`), all with the same locations, years and seed:
A = the original setup (`sowing_scheme: legacy`, run by `generate_wofost_dataset`), A_w = worlds with
only their sowing shift, B = A_w + cultivar, C = B + crop-parameter factors, D = C + irrigation,
E = D + extra locations.

Writes into `output_dir`:
- `dataset_manifest.csv`: one row per run (the original manifest's columns + `world_id`,
  `world_index`, `cultivar`, `production`, `world_kept`); `process_wofost_dataset` reads it as before;
- `worlds.parquet`: the hidden settings per world (cultivar, production, factors, sowing shift, seed,
  per-year sowing offsets) and whether it was kept / why not;
- `world_rejections.csv`: rejected worlds per country and reason;
- `config.yaml`: the settings used.
"""

import argparse
import json
import multiprocessing
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from src.data_pipeline.soil.utils_soil.default_soil_variables import default_zs
from src.data_pipeline.wofost.generate_wofost_dataset import (
    _default_num_workers,
    _prefetch_weather_for_window,
    generate_wofost_dataset,
)
from src.data_pipeline.wofost.run_wofost_simulation import generate_wofost_episode
from src.data_pipeline.wofost.utils_wofost.pcse_runner import load_crop_data_provider
from src.data_pipeline.wofost.worlds import (
    WorldsConfig,
    load_irrigated_shares,
    load_locations,
    sample_worlds,
    save_config,
    scale_crop_parameters,
    selected,
    world_filter,
)

SOIL_DEPTH_CM = float(default_zs()[1][-1])  # bottom of the SoilGrids profile (200 cm)


def world_overrides(world: dict, config: WorldsConfig, base_cache: dict) -> dict:
    """The crop-parameter overrides of a world: its factors applied to its cultivar's parameters."""
    factors = {k[len("factor_"):]: v for k, v in world.items() if k.startswith("factor_")}
    if not factors:
        return {}
    if world["cultivar"] not in base_cache:
        crop_data = load_crop_data_provider(None, config.crop, world["cultivar"])
        base_cache[world["cultivar"]] = {name: crop_data[name] for name in config.factor_ranges}
    return scale_crop_parameters(base_cache[world["cultivar"]], factors, SOIL_DEPTH_CM, config.rdmcr_step_cm)


def _manifest_row(world: dict, location: dict, config: WorldsConfig, year: int) -> dict:
    return {
        "location_index": world["location_index"], "longitude": world["longitude"],
        "latitude": world["latitude"], "crop": config.crop, "year": year, "jitter_index": 0,
        "sowing_offset_days": world["sowing_offsets"][year], "sos_doy": location["sos_doy"],
        "eos_doy": location["eos_doy"], "world_id": world["world_id"], "world_index": world["world_index"],
        "cultivar": world["cultivar"], "production": world["production"], "status": "failed", "error": None,
        "sowing_date": None, "yield_kg_per_ha": None, "final_dvs": None, "flowering_date": None,
        "maturity_date": None, "runtime_s": None, "awc": None, "bulk_density": None, "daily_path": None,
        "summary_path": None,
    }


def run_location(location: dict, worlds: list, config: WorldsConfig) -> list:
    """Every year of every world of one location, one after the other; returns the manifest rows.
    Module-level so it can run in a worker process (one location per task: the soil/weather
    caches are one file per coordinate, so two workers never write the same file)."""
    location = {**location, "sos_doy": int(location["sos_doy"]), "eos_doy": int(location["eos_doy"])}
    max_offset = config.world_shift_days + config.year_jitter_days
    _prefetch_weather_for_window(location["longitude"], location["latitude"], config.crop, config.years,
                                 max_offset, sos_doy=location["sos_doy"], eos_doy=location["eos_doy"])
    base_cache, rows = {}, []
    for world in worlds:
        overrides = world_overrides(world, config, base_cache)
        for year in config.years:
            row = _manifest_row(world, location, config, year)
            tag = f"{world['world_id']} {year} ({config.name})"
            start = time.perf_counter()
            try:
                result = generate_wofost_episode(
                    longitude=world["longitude"], latitude=world["latitude"], crop=config.crop, year=year,
                    variety_name=world["cultivar"], sowing_doy=location["sos_doy"], eos_doy=location["eos_doy"],
                    sowing_offset_days=world["sowing_offsets"][year], param_overrides=overrides,
                    production=world["production"], file_tag=world["world_id"], output_dir=config.output_dir,
                )
                summary = result["summary"]
                row.update(status="success", **{k: summary.get(k) for k in (
                    "sowing_date", "yield_kg_per_ha", "final_dvs", "flowering_date", "maturity_date", "awc",
                    "bulk_density")}, daily_path=result["daily_path"], summary_path=result["summary_path"])
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"
                print(f"{tag} FAILED: {row['error']}", flush=True)
                traceback.print_exc()
            row["runtime_s"] = round(time.perf_counter() - start, 3)
            rows.append(row)
        print(f"{world['world_id']} done ({config.name})", flush=True)
    return rows


def _legacy_worlds(manifest: pd.DataFrame, locations: pd.DataFrame, config: WorldsConfig) -> pd.DataFrame:
    """Dataset A: one world per location (the default cultivar, water-limited, no factors)."""
    worlds = []
    for loc, runs in manifest.groupby("location_index"):
        location = locations.loc[locations["location_index"] == loc].iloc[0]
        worlds.append({"world_id": f"L{int(loc):05d}_W00", "world_index": 0, "location_index": int(loc),
                       "longitude": location["longitude"], "latitude": location["latitude"],
                       "region": str(location.get(config.irrigation_region_column, "")),
                       "cultivar": config.default_cultivar, "production": "WLP", "irrigated_share": 0.0,
                       "sowing_shift_days": 0, "seed": config.seed})
    return pd.DataFrame(worlds)


def apply_filter(manifest: pd.DataFrame, worlds: pd.DataFrame, config: WorldsConfig) -> tuple:
    """Mark every world kept or rejected (`world_kept` in the manifest, `kept`/`reject_reasons` and the
    fractions per reason in `worlds`)."""
    verdicts = {}
    for world_id, runs in manifest.groupby("world_id"):
        if config.sowing_scheme == "legacy":  # judge A on one jitter per year, like a world
            runs = runs[runs["jitter_index"] == 0]
        verdicts[world_id] = world_filter(runs, config)
    worlds["kept"] = worlds["world_id"].map(lambda w: verdicts[w][0])
    worlds["reject_reasons"] = worlds["world_id"].map(lambda w: ",".join(verdicts[w][1]))
    for reason in ("failed", "immature", "low_yield"):
        worlds[f"frac_{reason}"] = worlds["world_id"].map(lambda w: verdicts[w][2][reason])
    manifest["world_kept"] = manifest["world_id"].map(worlds.set_index("world_id")["kept"])
    return manifest, worlds


def rejection_report(worlds: pd.DataFrame) -> pd.DataFrame:
    """Per region (country): worlds, rejected share, and the share rejected for each reason."""
    rows = []
    for region, g in worlds.groupby("region"):
        row = {"region": region, "worlds": len(g), "rejected": int((~g["kept"]).sum()),
               "rejection_rate": float((~g["kept"]).mean())}
        for reason in ("failed", "immature", "low_yield"):
            row[f"rate_{reason}"] = float(g["reject_reasons"].str.contains(reason).mean())
        rows.append(row)
    report = pd.DataFrame(rows)
    total = {"region": "ALL", "worlds": len(worlds), "rejected": int((~worlds["kept"]).sum()),
             "rejection_rate": float((~worlds["kept"]).mean()),
             **{f"rate_{r}": float(worlds["reject_reasons"].str.contains(r).mean())
                for r in ("failed", "immature", "low_yield")}}
    return pd.concat([report, pd.DataFrame([total])], ignore_index=True)


def generate_worlds_dataset(config: WorldsConfig) -> dict:
    os.makedirs(config.output_dir, exist_ok=True)
    save_config(config, os.path.join(config.output_dir, "config.yaml"))
    locations = load_locations(config)
    todo = selected(locations, config)
    manifest_path = os.path.join(config.output_dir, "dataset_manifest.csv")
    start = time.perf_counter()

    if config.sowing_scheme == "legacy":
        generate_wofost_dataset(
            locations=locations.to_dict(orient="records"), crop=config.crop, start_year=config.start_year,
            end_year=config.end_year, num_jitters=config.num_jitters, sowing_jitter_days=config.sowing_jitter_days,
            seed=config.seed, output_dir=config.output_dir, num_workers=config.workers,
            max_locations=config.max_locations, location_indices=config.location_indices)
        manifest = pd.read_csv(manifest_path)
        manifest["world_id"] = manifest["location_index"].map(lambda i: f"L{int(i):05d}_W00")
        manifest["world_index"], manifest["cultivar"], manifest["production"] = 0, config.default_cultivar, "WLP"
        worlds = _legacy_worlds(manifest, locations, config)
    else:
        shares = load_irrigated_shares(config)
        jobs = [(loc, sample_worlds(loc, config, shares)) for loc in todo.to_dict(orient="records")]
        workers = config.workers or _default_num_workers()
        rows = []
        if workers <= 1:
            for loc, worlds in jobs:
                rows += run_location(loc, worlds, config)
        else:
            # spawn: each worker starts clean with its own Earth Engine client
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
                futures = [pool.submit(run_location, loc, worlds, config) for loc, worlds in jobs]
                for future in as_completed(futures):
                    rows += future.result()
        manifest = pd.DataFrame(rows).sort_values(["location_index", "world_index", "year"]).reset_index(drop=True)
        worlds = pd.DataFrame([w for _, ws in jobs for w in ws])
        worlds["sowing_offsets"] = worlds["sowing_offsets"].map(lambda d: json.dumps({str(k): v for k, v in d.items()}))

    manifest, worlds = apply_filter(manifest, worlds, config)
    manifest.to_csv(manifest_path, index=False)
    worlds.to_parquet(os.path.join(config.output_dir, "worlds.parquet"), index=False)
    report = rejection_report(worlds)
    report.to_csv(os.path.join(config.output_dir, "world_rejections.csv"), index=False)

    ok = manifest["status"] == "success"
    print(f"{config.name}: {len(worlds)} worlds, {len(manifest)} runs ({int(ok.sum())} succeeded) in "
          f"{time.perf_counter() - start:.0f} s; {int((~worlds['kept']).sum())} worlds rejected.")
    print(report.to_string(index=False, float_format="%.3f"))
    return {"manifest_path": manifest_path, "manifest": manifest, "worlds": worlds}


def main():
    p = argparse.ArgumentParser(description="Generate a WOFOST dataset of sampled worlds from a config.")
    p.add_argument("--config", required=True, help="configs/wofost_worlds/dataset_*.yaml")
    p.add_argument("--location-indices", nargs="+", type=int, default=None, help="Only these locations (rows of the list).")
    p.add_argument("--max-locations", type=int, default=None)
    p.add_argument("--worlds-per-location", type=int, default=None)
    p.add_argument("--start-year", type=int, default=None)
    p.add_argument("--end-year", type=int, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("-o", "--output-dir", default=None)
    args = p.parse_args()
    config = WorldsConfig.from_yaml(args.config, location_indices=args.location_indices,
                                    max_locations=args.max_locations, worlds_per_location=args.worlds_per_location,
                                    start_year=args.start_year, end_year=args.end_year, workers=args.workers,
                                    output_dir=args.output_dir)
    generate_worlds_dataset(config)


if __name__ == "__main__":
    main()
