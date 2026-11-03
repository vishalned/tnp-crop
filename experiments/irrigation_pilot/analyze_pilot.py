"""Irrigation pilot results: rainfed vs fully irrigated simulated yield against CY-Bench's reported yield.

    uv run python -m experiments.irrigation_pilot.analyze_pilot [--seasons <csv>]

Per country (ES, IT, DE, PL): mean yield of the rainfed run (old soil and OM fix), of the fully irrigated run (potential
production) and CY-Bench's mean regional yield 2005-2020; the irrigated share f that would match CY-Bench's mean yield
(`f x irrigated + (1 - f) x rainfed`, clipped to 0-1); and how well the year-to-year national yield tracks the reported one
(correlation of the yearly series; production / harvested area for CY-Bench) for 0, 25, 50, 75, 100 % irrigated.
Writes `pilot_per_country.csv`, `pilot_year_to_year.csv` and `figures/pilot.png` here.
"""

import argparse
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
CYBENCH = "/projects/prjs1788/CropBench/data/cybench/cybench-data/maize"
CODE = {"Spain": "ES", "Italy": "IT", "Germany": "DE", "Poland": "PL"}
YEARS = (2005, 2020)
FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)


def cybench_yields(code: str) -> tuple:
    c = pd.read_csv(f"{CYBENCH}/{code}/yield_maize_{code}.csv")
    c = c[c["harvest_year"].between(*YEARS) & (c["yield"] > 0)]
    if c["production"].notna().any():
        national = c.groupby("harvest_year")["production"].sum() / c.groupby("harvest_year")["harvest_area"].sum()
    else:  # DE has no production values: mean regional yield per year (as experiments/eu_grid_scaling/plot_data.py)
        national = c.groupby("harvest_year")["yield"].mean()
    return c["yield"].mean(), national


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", default="/projects/prjs1788/tnp-crop/irrigation_pilot/pilot_seasons.csv")
    args = p.parse_args()
    d = pd.read_csv(args.seasons)
    failed = d[d["error"].fillna("") != ""]
    d = d[d["error"].fillna("") == ""]
    d = d[(d.final_dvs_rainfed >= 2) & (d.final_dvs_pp >= 2)]  # both matured
    for col in ("rainfed", "pp", "old_soil"):
        d[col] = d[f"yield_{col}_kg_per_ha"] / 1000
    rows, yy = [], []
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for k, (country, code) in enumerate(CODE.items()):
        g = d[d.country == country]
        cyb_mean, cyb_nat = cybench_yields(code)
        rain, pp = g.rainfed.mean(), g.pp.mean()
        f = float(np.clip((cyb_mean - rain) / (pp - rain), 0, 1)) if pp > rain else np.nan
        row = {"country": country, "locations": g.location_index.nunique(), "seasons": len(g), "failed_seasons": int((failed.country == country).sum()),
               "old_soil_rainfed": g.old_soil.mean(), "rainfed": rain, "irrigated_pp": pp, "cybench": cyb_mean,
               "irrigated_share_matching_cybench": f, "irrigation_gain_pct": 100 * (pp / rain - 1),
               "sd_rainfed": g.rainfed.std(), "sd_pp": g.pp.std()}
        by_year = g.groupby("year")[["rainfed", "pp"]].mean()
        common = by_year.index.intersection(cyb_nat.index)
        for frac in FRACTIONS:
            mix = frac * by_year.pp + (1 - frac) * by_year.rainfed
            r = np.corrcoef(mix.loc[common], cyb_nat.loc[common])[0, 1]
            yy.append({"country": country, "irrigated_share": frac, "year_to_year_r": r, "cv_pct": 100 * mix.std() / mix.mean()})
            row[f"year_to_year_r_{int(frac * 100)}pct"] = r
        rows.append(row)
        ax = axes[0, k]
        ax.bar(["old soil\nrainfed", "OM fix\nrainfed", "irrigated\n(PP)", "CY-Bench"], [row["old_soil_rainfed"], rain, pp, cyb_mean],
               color=["#bbbbbb", "#7a7a7a", "#1b9e77", "#d95f02"])
        ax.set_title(country)
        ax.set_ylabel("mean yield, t/ha")
        ax = axes[1, k]
        ax.plot(cyb_nat.index, 100 * (cyb_nat / cyb_nat.mean() - 1), color="#d95f02", lw=2, label="CY-Bench")
        ax.plot(by_year.index, 100 * (by_year.rainfed / by_year.rainfed.mean() - 1), color="#7a7a7a", lw=1.5, label="rainfed")
        ax.plot(by_year.index, 100 * (by_year.pp / by_year.pp.mean() - 1), color="#1b9e77", lw=1.5, label="irrigated")
        ax.set_ylabel("% from mean")
    axes[1, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Irrigation pilot: fully irrigated (potential production) vs rainfed, 2005-2020 (top: mean yield; bottom: national year-to-year)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/figures/pilot.png", dpi=140)
    res = pd.DataFrame(rows)
    res.round(3).to_csv(f"{OUT}/pilot_per_country.csv", index=False)
    pd.DataFrame(yy).round(3).to_csv(f"{OUT}/pilot_year_to_year.csv", index=False)
    print(res.round(2).T.to_string())
    print(pd.DataFrame(yy).pivot(index="country", columns="irrigated_share", values="year_to_year_r").round(2).to_string())
    print(f"failed seasons (not used): {len(failed)}")


if __name__ == "__main__":
    main()
