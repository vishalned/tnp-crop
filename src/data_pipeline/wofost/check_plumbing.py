"""Spec step 1 plumbing check.

Runs `Wofost81_PP` (potential production, no soil/water balance at all) on a
location end to end: weather -> crop params -> agromanagement -> PCSE run ->
output extraction. This is purely a wiring sanity check -- its output is NOT
real pretraining data, since potential production has no soil-driven yield
variation, which is the entire point of the CYBench yield task. Once this
passes, use `run_wofost_simulation.generate_wofost_episode` (the real,
`Wofost81_WLP_MLWB`-based generator) instead.
"""

import argparse
import datetime
import random
import sys
from typing import Optional

import pcse.models as pcse_models
import rootutils

from src.data_pipeline.weather.utils_weather.gee_weather import get_gee_weather_provider_for_location
from src.data_pipeline.wofost.utils_wofost.agromanagement import build_agromanagement, jitter_sowing_date
from src.data_pipeline.wofost.utils_wofost.default_wofost_variables import (
    default_crop_variety,
    default_max_duration_days,
    default_sowing_doy,
    plumbing_check_model_name,
)
from src.data_pipeline.wofost.utils_wofost.pcse_runner import build_parameter_provider, run_wofost


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def check_plumbing(
    longitude: float,
    latitude: float,
    crop: str,
    year: int,
    variety_name: Optional[str] = None,
    sowing_doy: Optional[int] = None,
    sowing_jitter_days: int = 10,
    seed: Optional[int] = None,
) -> dict:
    variety_name = variety_name if variety_name is not None else default_crop_variety()[crop]
    anchor_doy = sowing_doy if sowing_doy is not None else default_sowing_doy()[crop]
    max_duration = default_max_duration_days()[crop]

    rng = random.Random(seed)
    sowing_date = jitter_sowing_date(year, anchor_doy, sowing_jitter_days, rng=rng)

    model_class = getattr(pcse_models, plumbing_check_model_name())

    print(f"getting weather for longitude: {longitude}, latitude: {latitude}, from {sowing_date}")
    weather_data_provider = get_gee_weather_provider_for_location(
        latitude=latitude,
        longitude=longitude,
        start_date=sowing_date,
        end_date=sowing_date + datetime.timedelta(days=max_duration),
    )

    params = build_parameter_provider(
        model_class=model_class,
        crop_name=crop,
        variety_name=variety_name,
        soil_data=None,  # DummySoilDataProvider: PP doesn't touch the water balance
    )
    agromanagement = build_agromanagement(
        crop_name=crop,
        variety_name=variety_name,
        sowing_date=sowing_date,
        max_duration=max_duration,
    )

    daily_output, summary = run_wofost(model_class, params, weather_data_provider, agromanagement)

    print(f"Plumbing check OK: {len(daily_output)} days simulated, terminal summary: {summary}")
    return summary


def main():
    if len(sys.argv) == 1:
        print("No arguments provided!")
        print(
            "Usage: python check_plumbing.py --lon <longitude> --lat <latitude> "
            "--crop <wheat|maize> --year <year> [--seed <int>]"
        )
        print("Example: python check_plumbing.py -lon 6.656 -lat 52.966 --crop wheat --year 2020")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Wofost81_PP plumbing check (not real pretraining data).")
    parser.add_argument("-lon", "--longitude", dest="longitude", type=float, required=True)
    parser.add_argument("-lat", "--latitude", dest="latitude", type=float, required=True)
    parser.add_argument("--crop", dest="crop", type=str, required=True, choices=["wheat", "maize"])
    parser.add_argument("--year", dest="year", type=int, required=True)
    parser.add_argument("--seed", dest="seed", type=int, default=None)

    args = parser.parse_args()

    check_plumbing(
        longitude=args.longitude,
        latitude=args.latitude,
        crop=args.crop,
        year=args.year,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
