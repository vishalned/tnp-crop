"""Irrigation pilot: fully irrigated (WOFOST potential production, `Wofost81_PP`) vs the rainfed run of the OM-fix table.

    uv run python -m analysis.irrigation_pilot.run_pilot --n-per-country 75 --out <csv>      # a CPU job

For N random locations per country (ES, IT, DE, PL) that simulated all 48 seasons in the OM-fix run, every season is rerun
as potential production (no water limitation, no soil: PP needs no soil input) with the SAME weather, sowing date, crop
calendar and cultivar, so each season has a rainfed (OM-fix manifest) and a fully irrigated yield. Weather comes from
the caches only (a missing file is an error, never a download). Output: one row per season.
"""

import argparse
import datetime
import multiprocessing as mp
import os

os.environ.setdefault("TNP_NO_GEE_WEATHER", "1")

import pandas as pd  # noqa: E402
import pcse.models as pcse_models  # noqa: E402

from src.data_pipeline.weather.utils_weather.gee_weather import get_gee_weather_provider_for_location  # noqa: E402
from src.data_pipeline.wofost.utils_wofost.agromanagement import build_agromanagement, sowing_date_from_offset  # noqa: E402
from src.data_pipeline.wofost.utils_wofost.default_wofost_variables import (  # noqa: E402
    default_crop_variety,
    default_max_duration_days,
    default_pre_season_days,
)
from src.data_pipeline.wofost.utils_wofost.pcse_runner import build_parameter_provider, run_wofost  # noqa: E402

RUN = "/projects/prjs1788/tnp-crop"
OMFIX = f"{RUN}/wofost_eu_grid_dense_omfix/dataset_manifest.csv"
OLD = f"{RUN}/wofost_eu_grid_dense/dataset_manifest.csv"
LOCATIONS = "data/raw/locations/locations_maize_eu27_dense700_s42.csv"
COUNTRIES = ["Spain", "Italy", "Germany", "Poland"]
CROP = "maize"


def run_season(row: dict) -> dict:
    """Potential production for one season of the manifest (the logic of `generate_wofost_episode`, without soil)."""
    year, anchor_doy, eos_doy = int(row["year"]), int(row["sos_doy"]), int(row["eos_doy"])
    max_duration = default_max_duration_days()[CROP]
    sowing_date = sowing_date_from_offset(year, anchor_doy, int(row["sowing_offset_days"]))
    season_start = sowing_date_from_offset(year, anchor_doy, 0)
    campaign_start = min(season_start - datetime.timedelta(days=default_pre_season_days()), sowing_date)
    campaign_end = sowing_date + datetime.timedelta(days=max_duration + 1)
    eos_year = year + 1 if eos_doy < anchor_doy else year
    season_end = datetime.date(eos_year, 1, 1) + datetime.timedelta(days=eos_doy - 1)
    campaign_end = max(campaign_end, season_end + datetime.timedelta(days=1))

    model_class = pcse_models.Wofost81_PP
    weather = get_gee_weather_provider_for_location(latitude=row["latitude"], longitude=row["longitude"],
                                                    start_date=campaign_start, end_date=campaign_end)
    variety = default_crop_variety()[CROP]
    params = build_parameter_provider(model_class, CROP, variety, soil_data=None)
    agro = build_agromanagement(CROP, variety, sowing_date, max_duration, campaign_start, campaign_end)
    _, summary = run_wofost(model_class, params, weather, agro)
    return {"yield_pp_kg_per_ha": summary.get("TWSO"), "final_dvs_pp": summary.get("DVS"), "sowing_date": sowing_date.isoformat()}


def run_location(rows: list) -> list:
    out = []
    for row in rows:
        try:
            out.append({**{k: row[k] for k in ("location_index", "country", "year", "jitter_index", "sowing_offset_days")},
                        "yield_rainfed_kg_per_ha": row["yield_kg_per_ha"], "final_dvs_rainfed": row["final_dvs"],
                        "yield_old_soil_kg_per_ha": row["yield_old"], **run_season(row), "error": ""})
        except Exception as e:  # a failed season is kept, with its error
            out.append({**{k: row[k] for k in ("location_index", "country", "year", "jitter_index")}, "error": f"{type(e).__name__}: {e}"[:200]})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n-per-country", type=int, default=75)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    loc = pd.read_csv(LOCATIONS, usecols=["country"]).rename_axis("location_index").reset_index()
    m = pd.read_csv(OMFIX, low_memory=False)
    old = pd.read_csv(OLD, low_memory=False, usecols=["location_index", "year", "jitter_index", "yield_kg_per_ha"]).rename(
        columns={"yield_kg_per_ha": "yield_old"})
    m = m.merge(loc, on="location_index").merge(old, on=["location_index", "year", "jitter_index"], how="left")
    ok = m.groupby("location_index")["status"].apply(lambda s: (s == "success").all())
    m = m[m["country"].isin(COUNTRIES) & m["location_index"].isin(ok[ok].index)]
    chosen = []
    for c in COUNTRIES:
        ids = sorted(m.loc[m["country"] == c, "location_index"].unique())
        chosen += list(pd.Series(ids).sample(min(args.n_per_country, len(ids)), random_state=args.seed))
    m = m[m["location_index"].isin(chosen)]
    jobs = [g.to_dict("records") for _, g in m.groupby("location_index")]
    print(f"{len(chosen)} locations, {len(m)} seasons", flush=True)

    with mp.Pool(args.workers) as pool:
        rows = [r for part in pool.imap_unordered(run_location, jobs) for r in part]
    df = pd.DataFrame(rows).sort_values(["country", "location_index", "year", "jitter_index"])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"{len(df)} seasons, {(df['error'] != '').sum()} failed -> {args.out}", flush=True)
    print(df.loc[df["error"] != "", "error"].str.slice(0, 80).value_counts().head(5).to_string(), flush=True)


if __name__ == "__main__":
    main()
