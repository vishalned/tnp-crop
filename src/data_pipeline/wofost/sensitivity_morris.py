"""Morris sensitivity analysis of the crop-parameter factors (rain-fed), to pick which to keep.

    uv run python -m src.data_pipeline.wofost.sensitivity_morris --config configs/wofost_worlds/dataset_C.yaml \
        --out /projects/prjs1788/tnp-crop/wofost_worlds/sensitivity [--n-locations 5] [--years 2008 2012 2018] \
        [--trajectories 10] [--workers 8]

Parameters and ranges = the config's `factor_ranges` (the multiplicative factors of `worlds.py`),
cultivar = its `default_cultivar`, water-limited production. Locations: `--location-indices`, or
`--n-locations` spread evenly over the latitude range of the selected locations (north to south).
For every (location, year) one Morris design ((k + 1) x trajectories runs, k = number of parameters)
is simulated and analysed (SALib) for three outputs: yield (TWSO, t/ha), days from sowing to
flowering (anthesis) and to maturity (the crop's max duration if not reached). Writes:

- `morris_per_site_year.csv`: mu*, sigma and mu* confidence per parameter, output, location, year;
- `morris_summary.csv`: mean mu* and sigma over locations/years, and mu* relative to the largest
  parameter of that output (0-1);
- `morris_mu_star.png`: bar plot of mean mu* per output.
"""

import argparse
import datetime
import os
from concurrent.futures import ProcessPoolExecutor

import matplotlib
import numpy as np
import pandas as pd
import pcse.models as pcse_models

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from SALib.analyze import morris as morris_analyze  # noqa: E402
from SALib.sample import morris as morris_sample  # noqa: E402

from src.data_pipeline.soil.generate_gee_soil_file import generate_soil_data_for_wofost  # noqa: E402
from src.data_pipeline.weather.utils_weather.gee_weather import get_gee_weather_provider_for_location  # noqa: E402
from src.data_pipeline.wofost.generate_worlds_dataset import SOIL_DEPTH_CM  # noqa: E402
from src.data_pipeline.wofost.utils_wofost.agromanagement import build_agromanagement, sowing_date_from_offset  # noqa: E402
from src.data_pipeline.wofost.utils_wofost.default_wofost_variables import (  # noqa: E402
    default_max_duration_days,
    default_pre_season_days,
    wofost_model_name,
)
from src.data_pipeline.wofost.utils_wofost.pcse_runner import (  # noqa: E402
    build_parameter_provider,
    load_crop_data_provider,
    run_wofost,
)
from src.data_pipeline.wofost.worlds import WorldsConfig, load_locations, scale_crop_parameters, selected  # noqa: E402

OUTPUTS = ["yield_t_ha", "days_to_flowering", "days_to_maturity"]


def pick_locations(locations: pd.DataFrame, n: int) -> pd.DataFrame:
    """`n` locations spread evenly over the latitude range (north to south)."""
    ordered = locations.sort_values("latitude").reset_index(drop=True)
    return ordered.iloc[np.unique(np.linspace(0, len(ordered) - 1, n).round().astype(int))]


def simulate(location: dict, year: int, config: WorldsConfig, overrides: dict) -> dict:
    """One rain-fed season at the calendar's SOS with the given parameter overrides (no files written)."""
    crop, sos, eos = config.crop, int(location["sos_doy"]), int(location["eos_doy"])
    max_duration = default_max_duration_days()[crop]
    sowing = sowing_date_from_offset(year, sos, 0)
    start = sowing - datetime.timedelta(days=default_pre_season_days())
    eos_year = year + 1 if eos < sos else year
    end = max(sowing + datetime.timedelta(days=max_duration + 1),
              datetime.date(eos_year, 1, 1) + datetime.timedelta(days=eos))
    soil, _ = generate_soil_data_for_wofost(location["longitude"], location["latitude"])
    weather = get_gee_weather_provider_for_location(latitude=location["latitude"], longitude=location["longitude"],
                                                    start_date=start, end_date=end)
    model = getattr(pcse_models, wofost_model_name())
    params = build_parameter_provider(model, crop, config.default_cultivar, soil, param_overrides=overrides)
    agro = build_agromanagement(crop, config.default_cultivar, sowing, max_duration, campaign_start=start,
                                campaign_end=end)
    _, summary = run_wofost(model, params, weather, agro)

    def days(key):
        day = summary.get(key)
        return (day - sowing).days if isinstance(day, datetime.date) else max_duration

    return {"yield_t_ha": (summary.get("TWSO") or 0.0) / 1000.0, "days_to_flowering": days("DOA"),
            "days_to_maturity": days("DOM")}


def run_site_year(location: dict, year: int, config: WorldsConfig, problem: dict, design: np.ndarray) -> list:
    crop_data = load_crop_data_provider(None, config.crop, config.default_cultivar)
    base = {name: crop_data[name] for name in problem["names"]}
    results = []
    for row in design:
        overrides = scale_crop_parameters(base, dict(zip(problem["names"], row)), SOIL_DEPTH_CM, config.rdmcr_step_cm)
        results.append(simulate(location, year, config, overrides))
    y = pd.DataFrame(results)
    rows = []
    for output in OUTPUTS:
        si = morris_analyze.analyze(problem, design, y[output].to_numpy(float), num_levels=4)
        for i, name in enumerate(problem["names"]):
            rows.append({"location_index": int(location["location_index"]), "country": location.get("country"),
                         "latitude": location["latitude"], "year": year, "output": output, "parameter": name,
                         "mu_star": float(si["mu_star"][i]), "sigma": float(si["sigma"][i]),
                         "mu_star_conf": float(si["mu_star_conf"][i]), "output_mean": float(y[output].mean()),
                         "output_std": float(y[output].std())})
    print(f"location {location['location_index']} year {year}: {len(design)} runs done", flush=True)
    return rows


def bar_plot(summary: pd.DataFrame, path: str) -> None:
    labels = {"yield_t_ha": "yield (t/ha)", "days_to_flowering": "sowing → flowering (days)",
              "days_to_maturity": "sowing → maturity (days)"}
    fig, axes = plt.subplots(1, len(OUTPUTS), figsize=(11, 3.2))
    for ax, output in zip(axes, OUTPUTS):
        d = summary[summary["output"] == output].sort_values("mu_star")
        ax.barh(d["parameter"], d["mu_star"], xerr=d["mu_star_sd"], color="#2a78d6", height=0.6,
                error_kw={"elinewidth": 1, "ecolor": "#52514e"})
        ax.set_title(labels[output], loc="left", fontsize=10)
        ax.set_xlabel("Morris μ* (mean |elementary effect|)")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle("Crop-parameter sensitivity (rain-fed; bars = mean over locations × years, ± sd)",
                 x=0.0, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Morris sensitivity of the crop-parameter factors.")
    p.add_argument("--config", default="configs/wofost_worlds/dataset_C.yaml")
    p.add_argument("--out", required=True)
    p.add_argument("--location-indices", nargs="+", type=int, default=None)
    p.add_argument("--n-locations", type=int, default=5)
    p.add_argument("--years", nargs="+", type=int, default=[2008, 2012, 2018])
    p.add_argument("--trajectories", type=int, default=10)
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args()

    config = WorldsConfig.from_yaml(args.config, location_indices=args.location_indices)
    locations = selected(load_locations(config), config)
    if args.location_indices is None:
        locations = pick_locations(locations, args.n_locations)
    names = list(config.factor_ranges)
    problem = {"num_vars": len(names), "names": names, "bounds": [config.factor_ranges[n] for n in names]}
    design = morris_sample.sample(problem, args.trajectories, num_levels=4, seed=config.seed)
    jobs = [(loc, year) for loc in locations.to_dict(orient="records") for year in args.years]
    print(f"{len(jobs)} location-years x {len(design)} runs = {len(jobs) * len(design)} runs", flush=True)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_site_year, loc, year, config, problem, design) for loc, year in jobs]
        rows = [r for f in futures for r in f.result()]

    os.makedirs(args.out, exist_ok=True)
    per = pd.DataFrame(rows)
    per.to_csv(os.path.join(args.out, "morris_per_site_year.csv"), index=False)
    g = per.groupby(["output", "parameter"])
    summary = g[["mu_star", "sigma"]].mean().join(g["mu_star"].std().rename("mu_star_sd")).reset_index()
    summary["mu_star_relative"] = summary["mu_star"] / summary.groupby("output")["mu_star"].transform("max")
    summary = summary.sort_values(["output", "mu_star"], ascending=[True, False])
    summary.to_csv(os.path.join(args.out, "morris_summary.csv"), index=False)
    bar_plot(summary, os.path.join(args.out, "morris_mu_star.png"))
    print(summary.to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
