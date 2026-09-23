"""A small synthetic training store with the exact layout
`build_training_store.py` writes, for testing the dataloader/model without
real WOFOST runs. Values are plausible in scale only; yield depends on the
season's weather and soil so the signal is learnable."""

import datetime
import json
import os

import numpy as np
import pandas as pd

from src.data.components.crop_vocab import WEATHER_VARIABLES

DEFAULT_POINTS_PER_COUNTRY = {
    "France": 30, "Germany": 25, "United Kingdom": 10, "Netherlands": 6, "Denmark": 5, "Belgium": 4,
}


def make_synthetic_store(
    out_dir: str,
    points_per_country: dict = None,
    years: tuple = (2005, 2020),
    num_jitters: int = 3,
    season_start_doy: int = 288,
    seed: int = 0,
) -> str:
    rng = np.random.default_rng(seed)
    points_per_country = points_per_country or DEFAULT_POINTS_PER_COUNTRY
    rows = []
    for country, n in points_per_country.items():
        lat0, lon0 = rng.uniform(46, 55), rng.uniform(-3, 12)
        for _ in range(n):
            rows.append({"country": country, "latitude": lat0 + rng.normal(0, 0.8), "longitude": lon0 + rng.normal(0, 0.8)})
    points = pd.DataFrame(rows)
    P = len(points)
    points.insert(0, "point_id", np.arange(P))
    points.insert(1, "location_index", np.arange(P))
    points["zarr_index"] = rng.integers(0, 10**6, P)
    points["crop"] = "wheat"
    for name, (lo, hi) in {"clay": (100, 450), "nitrogen": (80, 400), "ph": (50, 80), "soc": (50, 400)}.items():
        base = rng.uniform(lo, hi, P)
        for layer in range(3):
            points[f"{name}_{layer}"] = base * (1 - 0.15 * layer) + rng.normal(0, (hi - lo) * 0.03, P)
    points["water_holding_capacity"] = rng.uniform(0.15, 0.3, P)
    points["elevation"] = rng.uniform(0, 400, P)
    points["slope"] = rng.uniform(0, 8, P)

    first, last = years
    start = datetime.date(first, 1, 1) + datetime.timedelta(days=season_start_doy - 1 - 120)
    num_days = (datetime.date(last + 2, 1, 1) - start).days
    doy = np.array([(start + datetime.timedelta(days=d)).timetuple().tm_yday for d in range(num_days)])
    season = np.cos(2 * np.pi * (doy - 200) / 365.25)  # +1 mid-July
    weather = np.empty((P, num_days, len(WEATHER_VARIABLES)), dtype=np.float32)
    lat_effect = (points["latitude"].to_numpy() - 50)[:, None]
    anomaly = rng.normal(0, 1, (P, num_days)).cumsum(axis=1) * 0.05
    tavg = 10 + 8 * season[None] - 0.6 * lat_effect + anomaly + rng.normal(0, 2, (P, num_days))
    weather[..., 0] = tavg - 4  # tmin
    weather[..., 1] = tavg + 4  # tmax
    weather[..., 2] = np.maximum(rng.gamma(0.6, 3.5, (P, num_days)) - 0.5, 0)  # precip mm/day
    weather[..., 3] = np.maximum(10 + 8 * season[None] + rng.normal(0, 3, (P, num_days)), 0.5)  # MJ/m2/day
    weather[..., 4] = np.abs(3 + rng.normal(0, 1.2, (P, num_days)))  # wind
    weather[..., 5] = 6.1 * np.exp(17.27 * (tavg - 4) / (tavg - 4 + 237.3))  # vapour pressure
    weather[rng.integers(P), rng.integers(num_days, size=20)] = np.nan  # a few missing days

    seasons = []
    for p in range(P):
        for y in range(first, last + 1):
            ss = datetime.date(y, 1, 1) + datetime.timedelta(days=season_start_doy - 1)
            i0 = (ss - start).days
            heat = float(np.nanmean(weather[p, i0 + 200 : i0 + 290, 1]))
            rain = float(np.nansum(weather[p, i0 + 150 : i0 + 280, 2]))
            base_yield = 9 + 0.004 * rain - 0.25 * (heat - 22) + 8 * (points.at[p, "water_holding_capacity"] - 0.22)
            for j, offset in enumerate(sorted(rng.choice(np.arange(-10, 11), num_jitters, replace=False))):
                sow = ss + datetime.timedelta(days=int(offset))
                maturity_days = int(290 - 1.5 * (heat - 22) + 0.3 * offset + rng.normal(0, 3))
                flowering_days = int(maturity_days - 70 + rng.normal(0, 3))
                maturity = ss + datetime.timedelta(days=maturity_days)
                seasons.append({
                    "point_id": p, "season_year": y, "jitter_index": j,
                    "sowing_date": sow.isoformat(), "season_start": ss.isoformat(),
                    "flowering_date": (ss + datetime.timedelta(days=flowering_days)).isoformat(),
                    "maturity_date": maturity.isoformat(),
                    "yield_t_per_ha": max(base_yield - 0.03 * abs(offset) + rng.normal(0, 0.4), 1.0),
                    "reached_maturity": True,
                    "flowering_days": flowering_days, "maturity_days": maturity_days,
                    "harvest_year": maturity.year,
                })

    os.makedirs(out_dir, exist_ok=True)
    points.to_csv(os.path.join(out_dir, "points.csv"), index=False)
    pd.DataFrame(seasons).to_csv(os.path.join(out_dir, "seasons.csv"), index=False)
    np.save(os.path.join(out_dir, "weather.npy"), weather)
    with open(os.path.join(out_dir, "weather_meta.json"), "w") as f:
        json.dump({"start_date": start.isoformat(), "num_days": num_days, "variables": WEATHER_VARIABLES, "crop": "wheat",
                   "synthetic": True}, f, indent=2)
    return out_dir
