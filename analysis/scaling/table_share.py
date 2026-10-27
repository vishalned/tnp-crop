"""Share of training tables per country: replays `CropTables.sample_batch` (structured episodes).

    uv run python -m analysis.scaling.table_share old=<table.parquet> eu_grid=<table.parquet> ...

For every table: 2000 steps x batch 8 x 3 seeds, the same rule as `nanotabicl/crop_tables.py` (lead
country uniform; rows per point = 1 + 5..11 context years; n_points in [10, min(lead's points,
4096 // rows per point)]; the other tables from countries with >= n_points points; a table only counts if
the country has n_points points with the target year and enough earlier matured train years). Only
the location / year / maturity columns are read; no features. The draws use their own random numbers,
so the numbers are what training sees on average, not the exact tables of a run.

Writes `analysis/scaling/table_share.csv`, one row per (run, country):
- `locations`: points of the country in the table (regions with < 10 are dropped, as in training);
- `usable`: points that matured in all 12 train years;
- `lead_share`: share of batches the country leads (1 / number of countries, up to noise);
- `lead_built_share`: share of batches whose first table could be built from the lead country itself;
- `table_share`: share of all tables that come from the country;
- `rows_min` ... `rows_max`: rows (context + query) of its tables; `rows_max_possible` by the rule.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

MIN_POINTS, MAX_ROWS, MIN_CTX, MAX_CTX, TRAIN = 10, 4096, 5, 11, (2005, 2016)
STEPS, BATCH, SEEDS = 2000, 8, (0, 1, 2)
OUT = Path(__file__).resolve().parent / "table_share.csv"


def load(path):
    df = pd.read_parquet(path, columns=["location_index", "country", "sowing_year", "yield_t_per_ha", "reached_maturity"])
    df = df[df["reached_maturity"].astype(bool) & np.isfinite(df["yield_t_per_ha"])]
    country = df.groupby("location_index")["country"].first()
    regions = {c: country.index[country == c].to_numpy() for c in sorted(country.unique())}
    points = {c: p for c, p in regions.items() if len(p) >= MIN_POINTS}
    years = df.groupby("location_index")["sowing_year"].agg(set).to_dict()
    pool = [y for y in sorted(df["sowing_year"].unique()) if TRAIN[0] <= y <= TRAIN[1]]
    return points, years, pool


def replay(points, years, pool):
    countries = list(points)
    candidates = {}  # (country, n_context, T) -> points with T and >= n_context earlier train years

    def n_candidates(c, n_ctx, T):
        key = (c, n_ctx, T)
        if key not in candidates:
            before = {y for y in pool if y < T}
            candidates[key] = sum(1 for p in points[c] if T in years[p] and len(years[p] & before) >= n_ctx)
        return candidates[key]

    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        for _ in range(STEPS):
            lead = countries[rng.integers(len(countries))]
            rpp = int(rng.integers(MIN_CTX, MAX_CTX + 1)) + 1
            n = int(rng.integers(MIN_POINTS, max(min(len(points[lead]), MAX_ROWS // rpp), MIN_POINTS) + 1))
            eligible = [c for c in countries if len(points[c]) >= n]
            targets = [T for T in pool if sum(y < T for y in pool) >= rpp - 1]
            built = 0
            for attempt in range(1000):
                c = lead if attempt == 0 else eligible[rng.integers(len(eligible))]
                T = int(rng.choice(targets))
                if n_candidates(c, rpp - 1, T) >= n:
                    rows.append((lead, c, attempt == 0, n * rpp))
                    built += 1
                    if built == BATCH:
                        break
    return pd.DataFrame(rows, columns=["lead", "country", "is_lead_table", "rows"])


def summarize(name, points, years, pool, t):
    n_batches = STEPS * len(SEEDS)
    out = t.groupby("country")["rows"].agg(rows_min="min", rows_median="median", rows_mean="mean", rows_max="max")
    out = out.reindex(list(points))
    out.insert(0, "locations", [len(points[c]) for c in out.index])
    out.insert(1, "usable", [sum(len(years[p] & set(pool)) == len(pool) for p in points[c]) for c in out.index])
    out["rows_max_possible"] = [max(min(len(points[c]), MAX_ROWS // r) * r for r in range(MIN_CTX + 1, MAX_CTX + 2))
                                for c in out.index]
    out["lead_share"] = t.groupby("lead").size().reindex(out.index).fillna(0) / BATCH / n_batches
    out["lead_built_share"] = t[t["is_lead_table"]].groupby("country").size().reindex(out.index).fillna(0) / n_batches
    out["table_share"] = t.groupby("country").size().reindex(out.index).fillna(0) / len(t)
    out = out.reset_index().rename(columns={"index": "country"})
    out.insert(0, "run", name)
    total = {"run": name, "country": "ALL", "locations": out["locations"].sum(), "usable": out["usable"].sum(),
             "rows_min": t["rows"].min(), "rows_median": t["rows"].median(), "rows_mean": t["rows"].mean(),
             "rows_max": t["rows"].max(), "table_share": 1.0}
    return pd.concat([out.sort_values("locations", ascending=False), pd.DataFrame([total])], ignore_index=True)


def main():
    runs = dict(arg.split("=", 1) for arg in sys.argv[1:])
    frames = []
    for name, path in runs.items():
        points, years, pool = load(path)
        frames.append(summarize(name, points, years, pool, replay(points, years, pool)))
        print(frames[-1].round(3).to_string(index=False), flush=True)
    previous = pd.read_csv(OUT) if OUT.exists() else None
    out = pd.concat(frames, ignore_index=True)
    if previous is not None:  # keep other runs' rows
        out = pd.concat([previous[~previous["run"].isin(runs)], out], ignore_index=True)
    out.round(4).to_csv(OUT, index=False)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
