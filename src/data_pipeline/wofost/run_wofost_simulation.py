import argparse
import datetime
import json
import os
import random
import sys
from typing import Optional

import pcse.models as pcse_models
import rootutils

from src.data_pipeline.paths import RAW_DIR
from src.data_pipeline.soil.generate_gee_soil_file import generate_soil_data_for_wofost
from src.data_pipeline.weather.utils_weather.gee_weather import get_gee_weather_provider_for_location
from src.data_pipeline.wofost.utils_wofost.agromanagement import (
    build_agromanagement,
    jitter_sowing_date,
    sowing_date_from_offset,
)
from src.data_pipeline.wofost.utils_wofost.default_wofost_variables import (
    default_crop_variety,
    default_max_duration_days,
    default_pre_season_days,
    default_sowing_doy,
    wofost_model_name,
)
from src.data_pipeline.wofost.utils_wofost.pcse_runner import (
    build_parameter_provider,
    merge_weather_and_derive_features,
    run_wofost,
)


root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
DEFAULT_WOFOST_SAVE_DIR = os.path.join(RAW_DIR, "wofost")


def generate_wofost_episode(
    longitude: float,
    latitude: float,
    crop: str,
    year: int,
    variety_name: Optional[str] = None,
    sowing_doy: Optional[int] = None,
    sowing_jitter_days: int = 10,
    seed: Optional[int] = None,
    output_dir: Optional[str] = None,
    force_refresh_soil: bool = False,
    sowing_offset_days: Optional[int] = None,
    eos_doy: Optional[int] = None,
    pre_season_days: Optional[int] = None,
    write_summary: bool = True,
) -> dict:
    """Run one end-to-end WOFOST/PCSE simulation episode for a location/year
    and cache the daily driver+output trajectory (with CYBench-aligned
    derived features) and a yield summary to the data directory.

    v1 simplification spec: CYBench yield task only, `Wofost81_WLP_MLWB`
    (water-limited, multi-layer waterbalance, no nitrogen). Soil is a real
    per-location, depth-resolved profile derived from a GEE SoilGrids pull
    (`generate_gee_soil_file.generate_soil_data_for_wofost`), not a single
    collapsed bucket.

    :param sowing_doy: day-of-year sowing anchor: the crop calendar's start
        of season (WorldCereal SOS; the batch runner passes the location's
        `sos_doy`). Defaults to a rough, location-agnostic placeholder per
        crop (see `default_wofost_variables.default_sowing_doy`).
    :param eos_doy: the crop calendar's end of season (WorldCereal EOS). The
        simulation then runs until at least that day (and at least until the
        crop's max duration), so the soil water balance covers the whole
        CY-Bench feature window even after maturity.
    :param pre_season_days: the campaign starts this many days before the
        season start (the unjittered anchor), so the soil water balance runs
        on bare soil before sowing (default
        `default_wofost_variables.default_pre_season_days()`, 60 = CY-Bench's
        `sos-60` window start).
    :param sowing_offset_days: sow exactly this many days after (negative:
        before) the anchor instead of a random jitter; `sowing_jitter_days`
        and `seed` are then unused. The batch runner uses this to run
        several distinct sowing dates per location/year.
    :param force_refresh_soil: re-query Earth Engine for soil even if a soil
        file already exists for this (longitude, latitude). By default an
        existing soil file is reused as-is -- soil only depends on location,
        not on crop/year/sowing date, and GEE quota is the scarce resource.
        Weather (ERA5-Land from Earth Engine) is also cached per location:
        the whole period from `default_weather_start_date()` to the present
        is fetched in one request; see
        `gee_weather.get_gee_weather_provider_for_location`.
    """
    variety_name = variety_name if variety_name is not None else default_crop_variety()[crop]
    anchor_doy = sowing_doy if sowing_doy is not None else default_sowing_doy()[crop]
    max_duration = default_max_duration_days()[crop]
    pre_season_days = pre_season_days if pre_season_days is not None else default_pre_season_days()

    if sowing_offset_days is not None:
        sowing_date = sowing_date_from_offset(year, anchor_doy, sowing_offset_days)
    else:
        sowing_date = jitter_sowing_date(year, anchor_doy, sowing_jitter_days, rng=random.Random(seed))

    season_start = sowing_date_from_offset(year, anchor_doy, 0)
    campaign_start = min(season_start - datetime.timedelta(days=pre_season_days), sowing_date)
    campaign_end = sowing_date + datetime.timedelta(days=max_duration + 1)
    if eos_doy is not None:
        # EOS before SOS in the calendar = the season ends the following year
        eos_year = year + 1 if eos_doy < anchor_doy else year
        season_end = datetime.date(eos_year, 1, 1) + datetime.timedelta(days=int(eos_doy) - 1)
        campaign_end = max(campaign_end, season_end + datetime.timedelta(days=1))

    model_class = getattr(pcse_models, wofost_model_name())

    print(f"building soil profile for longitude: {longitude}, latitude: {latitude}")
    soil_data, static_features = generate_soil_data_for_wofost(
        longitude, latitude, force_refresh=force_refresh_soil
    )

    print(f"getting weather to run WOFOST for longitude: {longitude}, latitude: {latitude}, from {campaign_start}")
    weather_data_provider = get_gee_weather_provider_for_location(
        latitude=latitude,
        longitude=longitude,
        start_date=campaign_start,
        end_date=campaign_end,
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
        campaign_start=campaign_start,
        campaign_end=campaign_end,
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
        "sowing_offset_days": (sowing_date - season_start).days,
        "sos_doy": int(anchor_doy),
        "eos_doy": int(eos_doy) if eos_doy is not None else None,
        "campaign_start": campaign_start.isoformat(),
        "campaign_end": campaign_end.isoformat(),
        # yield: total weight storage organs (kg/ha) at maturity
        "yield_kg_per_ha": summary.get("TWSO"),
        "final_dvs": summary.get("DVS"),
        # awc/bulk_density: derived from the topsoil layer of the real
        # GEE-sourced, depth-resolved soil profile (see
        # soil_static_features.compute_topsoil_static_features).
        "awc": static_features["awc"],
        "bulk_density": static_features["bulk_density"],
    }
    print(f"WOFOST daily trajectory written to {daily_path}.")
    if write_summary:  # the batch runner skips it: the manifest has the same fields (one file less per season)
        with open(summary_path, "w") as f:
            json.dump(summary_record, f, indent=2)
        print(f"WOFOST summary written to {summary_path}.")
    else:
        summary_path = None

    return {"daily_path": daily_path, "summary_path": summary_path, "summary": summary_record}


def main():
    if len(sys.argv) == 1:
        print("No arguments provided!")
        print(
            "Usage: python run_wofost_simulation.py --lon <longitude> --lat <latitude> "
            "--crop <wheat|maize> --year <year> [--sowing-doy <doy>] "
            "[--sowing-jitter-days <days> | --sowing-offset-days <days>] "
            "[--seed <int>] [--output-dir <path>] [--force-refresh-soil]"
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
    parser.add_argument("--sowing-offset-days", dest="sowing_offset_days", type=int, default=None, help="Sow exactly this many days after the anchor (overrides the random jitter).")
    parser.add_argument("--eos-doy", dest="eos_doy", type=int, default=None, help="Crop calendar end of season (day of year); the simulation runs until at least then.")
    parser.add_argument("--pre-season-days", dest="pre_season_days", type=int, default=None, help="Start the campaign this many days before the season start (default 60).")
    parser.add_argument("--seed", dest="seed", type=int, default=None)
    parser.add_argument("-o", "--output-dir", dest="output_dir", type=str, default=DEFAULT_WOFOST_SAVE_DIR)
    parser.add_argument("--force-refresh-soil", dest="force_refresh_soil", action="store_true", help="Re-query Earth Engine for soil even if a soil file already exists for this location.")

    args = parser.parse_args()

    generate_wofost_episode(
        longitude=args.longitude,
        latitude=args.latitude,
        crop=args.crop,
        year=args.year,
        variety_name=args.variety_name,
        sowing_doy=args.sowing_doy,
        sowing_jitter_days=args.sowing_jitter_days,
        seed=args.seed,
        output_dir=args.output_dir,
        force_refresh_soil=args.force_refresh_soil,
        sowing_offset_days=args.sowing_offset_days,
        eos_doy=args.eos_doy,
        pre_season_days=args.pre_season_days,
    )


if __name__ == "__main__":
    main()
