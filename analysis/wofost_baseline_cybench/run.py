"""WOFOST baseline vs CY-Bench: how well do simulated yields, aggregated to CY-Bench regions, match the
reported region-year yields? Training years only.

    uv run python -m analysis.wofost_baseline_cybench.run eu_grid=<processed table> [fr_be_dk=<table> ...]

Steps, per WOFOST table and benchmark country:

1. **Regions.** CY-Bench gives a centroid and an area per region (`location_maize_<CC>.csv`), no
   boundaries. Each WOFOST point goes to the nearest region centroid of its own country (a Voronoi
   approximation of the regions), unless it is farther than `CAP` x the region's radius sqrt(area / pi).
2. **Region-year yield.** Mean of the simulated yields (all seasons, also those that didn't mature: that
   is the simulated harvest) over the region's points and sowing dates; regions with < `MIN_POINTS` points
   are left out. Reported: CY-Bench's `yield` (t/ha), harvest year = sowing year for maize.
3. **Years.** 2005 up to the year before the country's first benchmark test year (`TEST_START`: the
   benchmark's walk-forward test block = the last 5 years of the country's data), so nothing here looks
   at a year the benchmark tests on.
4. **Metrics** over the region-years with both values:
   - `bias` = mean simulated - mean reported (t/ha); `nrmse` = RMSE / mean reported (the simulated yield
     used directly as a prediction, as in the benchmark's NRMSE);
   - `spatial_r`: correlation across regions of their mean yields (same years for both);
   - `temporal_r`: per region with >= `MIN_YEARS` years, correlation over years after removing each
     series' linear trend (reported yields rise with technology, WOFOST has no trend); median over regions;
   - `anomaly_r`: the same detrended deviations, pooled over all the country's region-years.

Writes to `analysis/wofost_baseline_cybench/`: `<name>_region_years.csv`, `<name>_per_country.csv`, and the
figures `<name>_scatter.png` (region-years, simulated vs reported, per country) and `<name>_agreement.png`.

**As a baseline on the benchmark's test years** (`--test`): the simulated region-year yield, uncalibrated,
is the prediction for exactly the benchmark's walk-forward test rows (taken from the default TabICL run,
`REFERENCE_RUN`). Nothing is fitted, so the test years can be used. Every region gets a prediction: its
assigned points, or its `MIN_POINTS` nearest points in the country if it has fewer. Years WOFOST doesn't
cover (after 2020: DE 2021) get none. Written in the benchmark's layout to
`BASELINE_ROOT/maize_<CC>_wofost_eos_raw_<name>/walk_forward/<year>/0/test_preds.csv`, so the benchmark's
`summarize_exp.py` scores it exactly like the models (its `per_run.csv` there).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.plot_results import INK, INK2, REF, WOFOST  # noqa: F401 (sets the shared plot style)

CYBENCH = Path("/projects/prjs1788/CropBench/data/cybench/cybench-data/maize")
OUT = Path(__file__).resolve().parent
COUNTRIES = {"Austria": "AT", "Belgium": "BE", "Bulgaria": "BG", "Croatia": "HR", "Czechia": "CZ", "Denmark": "DK",
             "France": "FR", "Germany": "DE", "Greece": "EL", "Hungary": "HU", "Italy": "IT", "Lithuania": "LT",
             "Netherlands": "NL", "Poland": "PL", "Portugal": "PT", "Romania": "RO", "Slovakia": "SK", "Spain": "ES",
             "Sweden": "SE"}
# first walk-forward test year of the benchmark per country (cybench_output/<exp>/<run>/walk_forward/)
TEST_START = {"DE": 2017, "EL": 2015, "SK": 2014}
DEFAULT_TEST_START = 2016
BENCHMARK = ["NL", "FR", "DE", "PL", "ES", "IT", "AT", "BE", "BG", "CZ", "DK", "EL", "HR", "HU", "LT", "RO", "SE", "SK"]
FIRST_YEAR, CAP, MIN_POINTS, MIN_YEARS = 2005, 1.5, 3, 5
REFERENCE_RUN = Path("/projects/prjs1788/tnp-crop/cybench_output/exp_1_eu_grid")  # its default TabICL run gives the test rows
BASELINE_ROOT = Path("/projects/prjs1788/tnp-crop/cybench_output/wofost_baseline")


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def assign_regions(points: pd.DataFrame, regions: pd.DataFrame) -> pd.Series:
    """adm_id of the nearest region centroid within CAP x its radius, per point (NaN if none)."""
    d = haversine_km(points["latitude"].to_numpy()[:, None], points["longitude"].to_numpy()[:, None],
                     regions["latitude"].to_numpy()[None, :], regions["longitude"].to_numpy()[None, :])
    nearest = d.argmin(axis=1)
    radius = np.sqrt(regions["region_area"].to_numpy() / np.pi)
    ok = d[np.arange(len(points)), nearest] <= CAP * radius[nearest]
    return pd.Series(np.where(ok, regions["adm_id"].to_numpy()[nearest], None), index=points.index)


def detrended(df: pd.DataFrame, col: str) -> pd.Series:
    """Per region: the series minus its linear trend over years (NaN for regions with < MIN_YEARS years)."""
    out = pd.Series(np.nan, index=df.index)
    for _, g in df.groupby("adm_id"):
        if len(g) >= MIN_YEARS:
            slope, intercept = np.polyfit(g["year"], g[col], 1)
            out[g.index] = g[col] - (slope * g["year"] + intercept)
    return out


def compare(table_path: str) -> tuple:
    t = pd.read_parquet(table_path, columns=["location_index", "country", "latitude", "longitude", "year", "yield_t_per_ha"])
    t = t[np.isfinite(t["yield_t_per_ha"])]
    region_years, rows = [], []
    for country, code in COUNTRIES.items():
        if code not in BENCHMARK or not (t["country"] == country).any():
            continue
        last = TEST_START.get(code, DEFAULT_TEST_START) - 1
        sim = t[(t["country"] == country) & t["year"].between(FIRST_YEAR, last)]
        points = sim.groupby("location_index")[["latitude", "longitude"]].first()
        regions = pd.read_csv(CYBENCH / code / f"location_maize_{code}.csv")
        points["adm_id"] = assign_regions(points, regions)
        sim = sim.assign(adm_id=sim["location_index"].map(points["adm_id"])).dropna(subset=["adm_id"])
        n_points = points.groupby("adm_id").size()
        keep = n_points.index[n_points >= MIN_POINTS]
        s = (sim[sim["adm_id"].isin(keep)].groupby(["adm_id", "year"])["yield_t_per_ha"].mean().rename("simulated"))
        rep = pd.read_csv(CYBENCH / code / f"yield_maize_{code}.csv").rename(columns={"harvest_year": "year"})
        rep = rep[rep["yield"].gt(0) & rep["year"].between(FIRST_YEAR, last)].groupby(["adm_id", "year"])["yield"].mean()
        ry = pd.concat([s, rep.rename("reported")], axis=1, join="inner").reset_index()
        ry.insert(0, "country", code)
        ry["points"] = ry["adm_id"].map(n_points)
        region_years.append(ry)
        row = {"country": code, "years": f"{FIRST_YEAR}-{last}", "points": len(points),
               "points_assigned": int(points["adm_id"].notna().sum()), "cybench_regions": rep.index.get_level_values(0).nunique(),
               "regions": ry["adm_id"].nunique(), "region_years": len(ry)}
        if len(ry):
            err = ry["simulated"] - ry["reported"]
            means = ry.groupby("adm_id")[["simulated", "reported"]].mean()
            ry["sim_anom"], ry["rep_anom"] = detrended(ry, "simulated"), detrended(ry, "reported")
            per_region = ry.dropna(subset=["sim_anom"]).groupby("adm_id")[["sim_anom", "rep_anom"]].apply(
                lambda g: g["sim_anom"].corr(g["rep_anom"]))
            anom = ry.dropna(subset=["sim_anom"])
            row.update({"mean_simulated": ry["simulated"].mean(), "mean_reported": ry["reported"].mean(),
                        "bias": err.mean(), "nrmse": np.sqrt((err ** 2).mean()) / ry["reported"].mean(),
                        "spatial_r": means["simulated"].corr(means["reported"]) if len(means) >= 3 else np.nan,
                        "temporal_r": per_region.median() if len(per_region) else np.nan,
                        "anomaly_r": anom["sim_anom"].corr(anom["rep_anom"]) if len(anom) >= 10 else np.nan})
        rows.append(row)
    return pd.concat(region_years, ignore_index=True), pd.DataFrame(rows)


def test_predictions(table_path: str, name: str) -> pd.DataFrame:
    """Uncalibrated WOFOST yield for the benchmark's walk-forward test rows; written as test_preds.csv."""
    t = pd.read_parquet(table_path, columns=["location_index", "country", "latitude", "longitude", "year", "yield_t_per_ha"])
    t = t[np.isfinite(t["yield_t_per_ha"])]
    coverage = []
    for country, code in COUNTRIES.items():
        if code not in BENCHMARK or not (t["country"] == country).any():
            continue
        run = REFERENCE_RUN / f"maize_{code}_tabicl_eos_tabicl_default_single" / "walk_forward"
        rows = pd.concat([pd.read_csv(min(y.glob("*/test_preds.csv"))) for y in sorted(run.iterdir())
                          if y.is_dir() and any(y.glob("*/test_preds.csv"))], ignore_index=True)
        sim = t[t["country"] == country]
        points = sim.groupby("location_index")[["latitude", "longitude"]].first()
        regions = pd.read_csv(CYBENCH / code / f"location_maize_{code}.csv").set_index("adm_id")
        points["adm_id"] = assign_regions(points, regions.reset_index())
        members = {}
        for adm_id in rows["adm_id"].unique():
            own = points.index[points["adm_id"] == adm_id]
            if len(own) < MIN_POINTS:  # too few of its own: the nearest points of the country
                d = haversine_km(points["latitude"], points["longitude"], regions.loc[adm_id, "latitude"], regions.loc[adm_id, "longitude"])
                own = points.index[np.argsort(d.to_numpy())[:MIN_POINTS]]
            members[adm_id] = own
        per_year = sim.groupby(["location_index", "year"])["yield_t_per_ha"].mean().unstack()
        rows["preds"] = [per_year.loc[members[a], y].mean() if y in per_year.columns else np.nan
                         for a, y in zip(rows["adm_id"], rows["year"])]
        out = BASELINE_ROOT / f"maize_{code}_wofost_eos_raw_{name}" / "walk_forward"
        for year, g in rows.dropna(subset=["preds"]).groupby("year"):
            (out / str(year) / "0").mkdir(parents=True, exist_ok=True)
            g[["adm_id", "year", "targets", "preds"]].to_csv(out / str(year) / "0" / "test_preds.csv", index=False)
        own_ok = sum(len(points.index[points["adm_id"] == a]) >= MIN_POINTS for a in members)
        coverage.append({"country": code, "test_rows": len(rows), "predicted": int(rows["preds"].notna().sum()),
                         "regions": len(members), "regions_own_points": own_ok,
                         "regions_nearest_points": len(members) - own_ok})
    coverage = pd.DataFrame(coverage)
    coverage.to_csv(OUT / f"{name}_test_coverage.csv", index=False)
    return coverage


def plot_scatter(ry: pd.DataFrame, summary: pd.DataFrame, name: str):
    countries = summary.dropna(subset=["nrmse"]).sort_values("region_years", ascending=False)["country"].tolist()
    ncol = 6
    nrow = int(np.ceil(len(countries) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.3 * ncol, 2.3 * nrow + 0.6))
    hi = float(np.nanpercentile(ry[["simulated", "reported"]].to_numpy(), 99.5))
    for ax, code in zip(axes.flat, countries):
        g, s = ry[ry["country"] == code], summary.set_index("country").loc[code]
        ax.plot([0, hi], [0, hi], color=INK2, lw=0.8, zorder=1)
        ax.scatter(g["reported"], g["simulated"], s=6, color=WOFOST, alpha=0.5, linewidths=0, zorder=2)
        ax.set_xlim(0, hi), ax.set_ylim(0, hi)
        ax.set_title(f"{code}  ({s['regions']:.0f} reg., {s['region_years']:.0f} r-y)", loc="left", fontsize=8.5)
        ax.text(0.03, 0.97, f"bias {s['bias']:+.1f}\nanom. r {s['anomaly_r']:.2f}", transform=ax.transAxes,
                va="top", fontsize=7.5, color=INK2)
    for ax in list(axes.flat)[len(countries):]:
        ax.axis("off")
    fig.supxlabel("CY-Bench reported yield (t/ha)", fontsize=9, color=INK2)
    fig.supylabel("WOFOST simulated yield, region mean (t/ha)", fontsize=9, color=INK2)
    fig.suptitle(f"WOFOST ({name}) vs CY-Bench, region-years of training years only; line = 1:1",
                 x=0.0, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}_scatter.png")
    plt.close(fig)


def plot_agreement(summary: pd.DataFrame, name: str):
    s = summary.dropna(subset=["nrmse"]).sort_values("anomaly_r")
    y = np.arange(len(s))
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 0.32 * len(s) + 1.8), sharey=True,
                                   gridspec_kw={"width_ratios": [1.3, 1]})
    ax0.axvline(0, color=INK2, lw=1)
    for col, color, marker, label in (("anomaly_r", WOFOST, "o", "anomaly r (pooled, detrended)"),
                                      ("temporal_r", "#9ec5f4", "o", "temporal r (median over regions, detrended)"),
                                      ("spatial_r", REF, "D", "spatial r (region means)")):
        ax0.scatter(s[col], y, s=34, color=color, marker=marker, edgecolors="white", zorder=3, label=label)
    ax0.set_yticks(y, [f"{c} ({n:.0f} reg.)" for c, n in zip(s["country"], s["regions"])])
    ax0.set_xlabel("correlation, WOFOST vs CY-Bench")
    ax0.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1 - 2.0 / (0.32 * len(s) + 1.8) / 10), fontsize=7.5, ncol=1)
    ax0.grid(axis="y", visible=False)
    ax1.axvline(0, color=INK2, lw=1)
    ax1.barh(y, s["bias"], color=[WOFOST if b >= 0 else "#eb6834" for b in s["bias"]], height=0.6)
    for yi, (b, n) in enumerate(zip(s["bias"], s["nrmse"])):
        ax1.text(b, yi, f"  NRMSE {n:.2f}  " if b >= 0 else f"NRMSE {n:.2f}  ", va="center",
                 ha="left" if b >= 0 else "right", fontsize=7, color=INK2)
    ax1.set_xlabel("bias: simulated − reported (t/ha)")
    ax1.grid(axis="y", visible=False)
    lo, hi = ax1.get_xlim()
    ax1.set_xlim(lo - 0.35 * (hi - lo), hi + 0.35 * (hi - lo))
    fig.suptitle(f"WOFOST ({name}) vs CY-Bench per country, training years only", x=0.0, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}_agreement.png")
    plt.close(fig)


def main():
    test = "--test" in sys.argv
    for arg in [a for a in sys.argv[1:] if a != "--test"]:
        name, path = arg.split("=", 1)
        if test:
            print(f"== {name}: test rows\n{test_predictions(path, name).to_string(index=False)}")
            continue
        ry, summary = compare(path)
        ry.round(4).to_csv(OUT / f"{name}_region_years.csv", index=False)
        summary.round(4).to_csv(OUT / f"{name}_per_country.csv", index=False)
        plot_scatter(ry, summary, name)
        plot_agreement(summary, name)
        print(f"== {name}\n{summary.round(3).to_string(index=False)}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
