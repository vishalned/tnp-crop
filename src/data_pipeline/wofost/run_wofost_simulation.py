import argparse
import json
import os
import random
import sys
from typing import Optional

import pcse.models as pcse_models
import rootutils

from src.data_pipeline.soil.generate_gee_soil_file import generate_soil_data_for_wofost
from src.data_pipeline.weather.utils_weather.openmeteo_weather import request_openmeteo_weather
from src.data_pipeline.wofost.utils_wofost.agromanagement import build_agromanagement, jitter_sowing_date
from src.data_pipeline.wofost.utils_wofost.default_wofost_variables import (
    default_crop_variety,
    default_max_duration_days,
    default_sowing_doy,
    wofost_model_name,
)
from src.data_pipeline.wofost.utils_wofost.pcse_runner import (
    build_parameter_provider,
    merge_weather_and_derive_features,
    run_wofost,
)


root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
DEFAULT_WOFOST_SAVE_DIR = os.path.join(str(root), "data", "processed", "wofost")


def generate_wofost_episode(
    longitude: float,
    latitude: float,
    crop: str,
    year: int,
    variety_name: Optional[str] = None,
    sowing_doy: Optional[int] = None,
    sowing_jitter_days: int = 10,
    openmeteo_model: Optional[str] = None,
    seed: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> dict:
    """Run one end-to-end WOFOST/PCSE simulation episode for a location/year
    and cache the daily driver+output trajectory (with CYBench-aligned
    derived features) and a yield summary to the data directory.

    v1 simplification spec: CYBench yield task only, `Wofost81_WLP_MLWB`
    (water-limited, multi-layer waterbalance, no nitrogen). Soil is a real
    per-location, depth-resolved profile derived from a GEE SoilGrids pull
    (`generate_gee_soil_file.generate_soil_data_for_wofost`), not a single
    collapsed bucket.

    :param sowing_doy: day-of-year sowing anchor. Defaults to a rough,
        location-agnostic placeholder per crop (see
        `default_wofost_variables.default_sowing_doy`) -- TODO: replace with
        the real per-location WorldCereal start-of-season (SOS) once that
        extraction pipeline exists.
    """
    variety_name = variety_name if variety_name is not None else default_crop_variety()[crop]
    anchor_doy = sowing_doy if sowing_doy is not None else default_sowing_doy()[crop]
    max_duration = default_max_duration_days()[crop]

    rng = random.Random(seed)
    sowing_date = jitter_sowing_date(year, anchor_doy, sowing_jitter_days, rng=rng)

    model_class = getattr(pcse_models, wofost_model_name())

    print(f"building soil profile for longitude: {longitude}, latitude: {latitude}")
    soil_data, static_features = generate_soil_data_for_wofost(longitude, latitude)

    print(f"getting weather to run WOFOST for longitude: {longitude}, latitude: {latitude}, from {sowing_date}")
    weather_data_provider = request_openmeteo_weather(
        latitude=latitude,
        longitude=longitude,
        start_date=sowing_date,
        openmeteo_model=openmeteo_model,
    )

    params = build_parameter_provider(
        model_class=model_class,
        crop_name=crop,
        variety_name=variety_name,
        soil_data=soil_data,
    )
    agromanagement = build_agromanagement(
        crop_name=crop,
        variety_name=variety_name,
        sowing_date=sowing_date,
        max_duration=max_duration,
    )

    daily_output, summary = run_wofost(model_class, params, weather_data_provider, agromanagement)
    daily_output = merge_weather_and_derive_features(
        daily_output, weather_data_provider, latitude, longitude, params
    )

    save_dir = output_dir if output_dir is not None else DEFAULT_WOFOST_SAVE_DIR
    save_dir = os.path.join(save_dir, crop)
    os.makedirs(save_dir, exist_ok=True)

    file_stem = f"wofost_{crop}_{longitude}_{latitude}_{year}_{sowing_date.isoformat()}"
    daily_path = os.path.join(save_dir, f"{file_stem}.csv")
    summary_path = os.path.join(save_dir, f"{file_stem}_summary.json")

    daily_output.to_csv(daily_path, index=False)

    summary_record = {
        "longitude": longitude,
        "latitude": latitude,
        "crop": crop,
        "variety_name": variety_name,
        "year": year,
        "sowing_date": sowing_date.isoformat(),
        # yield: total weight storage organs (kg/ha) at maturity
        "yield_kg_per_ha": summary.get("TWSO"),
        "final_dvs": summary.get("DVS"),
        # awc/bulk_density: derived from the topsoil layer of the real
        # GEE-sourced, depth-resolved soil profile (see
        # soil_static_features.compute_topsoil_static_features).
        "awc": static_features["awc"],
        "bulk_density": static_features["bulk_density"],
    }
    with open(summary_path, "w") as f:
        json.dump(summary_record, f, indent=2)

    print(f"WOFOST daily trajectory written to {daily_path}.")
    print(f"WOFOST summary written to {summary_path}.")

    return {"daily_path": daily_path, "summary_path": summary_path, "summary": summary_record}


def main():
    if len(sys.argv) == 1:
        print("No arguments provided!")
        print(
            "Usage: python run_wofost_simulation.py --lon <longitude> --lat <latitude> "
            "--crop <wheat|maize> --year <year> [--sowing-doy <doy>] "
            "[--sowing-jitter-days <days>] [--openmeteo-model era5_land] "
            "[--seed <int>] [--output-dir <path>]"
        )
        print("Example: python run_wofost_simulation.py -lon 6.656 -lat 52.966 --crop wheat --year 2020")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Run a WOFOST/PCSE (Wofost81_WLP_MLWB) simulation episode for a location/year and cache the result.")
    parser.add_argument("-lon", "--longitude", dest="longitude", type=float, required=True)
    parser.add_argument("-lat", "--latitude", dest="latitude", type=float, required=True)
    parser.add_argument("--crop", dest="crop", type=str, required=True, choices=["wheat", "maize"])
    parser.add_argument("--year", dest="year", type=int, required=True)
    parser.add_argument("--variety-name", dest="variety_name", type=str, default=None)
    parser.add_argument("--sowing-doy", dest="sowing_doy", type=int, default=None)
    parser.add_argument("--sowing-jitter-days", dest="sowing_jitter_days", type=int, default=10)
    parser.add_argument("--openmeteo-model", dest="openmeteo_model", type=str, default=None)
    parser.add_argument("--seed", dest="seed", type=int, default=None)
    parser.add_argument("-o", "--output-dir", dest="output_dir", type=str, default=DEFAULT_WOFOST_SAVE_DIR)

    args = parser.parse_args()

    generate_wofost_episode(
        longitude=args.longitude,
        latitude=args.latitude,
        crop=args.crop,
        year=args.year,
        variety_name=args.variety_name,
        sowing_doy=args.sowing_doy,
        sowing_jitter_days=args.sowing_jitter_days,
        openmeteo_model=args.openmeteo_model,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
