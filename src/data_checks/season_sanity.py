"""Per-country sanity check of WOFOST runs: yields, near-zero yields, maturity, and the season's dates
against the crop calendar. Compares several runs (e.g. the FR/BE/DK run and the EU grid run).

    uv run python -m src.data_checks.season_sanity \
        --run old /projects/prjs1788/tnp-crop/wofost_calendar/dataset_manifest.csv data/raw/locations/locations_maize.csv \
        --run new /projects/prjs1788/tnp-crop/wofost_eu_grid/dataset_manifest.csv data/raw/locations/locations_maize_eu27_cell01_n2000_s42.csv \
        --out analysis/scaling/sanity --workers 16

Reads every successful season's daily output (DVS) to get flowering (first day DVS >= 1) and maturity
(first day DVS >= 2), so run it in a job. Writes `season_dates.csv.gz` (one row per season),
`sanity_per_country.csv` and `sanity_per_country.png`. Dates are days relative to the location's crop
calendar: sowing - SOS, flowering - SOS, maturity - EOS (EOS in the sowing year, or the next year when
EOS < SOS, as in the runner).
"""

import argparse
import datetime
import os
from concurrent.futures import ProcessPoolExecutor

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {"old": "#8a8984", "new": "#2a78d6"}
NEAR_ZERO_T_HA = 0.5


def _doy_date(year: int, doy: int) -> datetime.date:
    return datetime.date(year, 1, 1) + datetime.timedelta(days=int(doy) - 1)


def season_dates(row: dict) -> dict:
    daily = pd.read_csv(row["daily_path"], usecols=["day", "DVS"])
    day = pd.to_datetime(daily["day"]).dt.date
    dvs = daily["DVS"].to_numpy(float)

    def first(threshold):
        hit = np.flatnonzero(dvs >= threshold)
        return day.iloc[hit[0]] if len(hit) else None

    year, sos, eos = int(row["year"]), int(row["sos_doy"]), int(row["eos_doy"])
    sos_date = _doy_date(year, sos)
    eos_date = _doy_date(year + 1 if eos < sos else year, eos)
    sowing = pd.to_datetime(row["sowing_date"]).date()
    flowering, maturity = first(1.0), first(2.0)
    return {"sowing_minus_sos": (sowing - sos_date).days,
            "flowering_minus_sos": (flowering - sos_date).days if flowering else np.nan,
            "maturity_minus_eos": (maturity - eos_date).days if maturity else np.nan,
            "sowing_doy": sowing.timetuple().tm_yday,
            "flowering_doy": flowering.timetuple().tm_yday if flowering else np.nan,
            "maturity_doy": maturity.timetuple().tm_yday if maturity else np.nan,
            "reached_maturity": maturity is not None, "matured_by_eos": bool(maturity and maturity <= eos_date)}


def load_run(name: str, manifest: str, locations: str, workers: int) -> pd.DataFrame:
    m = pd.read_csv(manifest, float_precision="round_trip")
    m["country"] = m["location_index"].map(pd.read_csv(locations)["country"])
    ok = m[m["status"] == "success"].reset_index(drop=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        dates = list(pool.map(season_dates, ok.to_dict(orient="records"), chunksize=256))
    out = pd.concat([ok, pd.DataFrame(dates)], axis=1).assign(run=name, yield_t_ha=lambda d: d["yield_kg_per_ha"] / 1000)
    failed = m[m["status"] != "success"].assign(run=name)
    return pd.concat([out, failed], ignore_index=True)


def summarize(seasons: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (run, country), g in seasons.groupby(["run", "country"]):
        ok = g[g["status"] == "success"]
        q = lambda c, p: ok[c].quantile(p)  # noqa: E731
        rows.append({"run": run, "country": country, "locations": g["location_index"].nunique(),
                     "seasons": len(g), "failed_share": (g["status"] != "success").mean(),
                     "yield_p10": q("yield_t_ha", .1), "yield_p50": q("yield_t_ha", .5), "yield_p90": q("yield_t_ha", .9),
                     "near_zero_share": (ok["yield_t_ha"] < NEAR_ZERO_T_HA).mean(),
                     "reached_maturity_share": ok["reached_maturity"].mean(),
                     "matured_by_eos_share": ok["matured_by_eos"].mean(),
                     "sos_doy_median": ok["sos_doy"].median(), "eos_doy_median": ok["eos_doy"].median(),
                     "sowing_minus_sos_median": q("sowing_minus_sos", .5),
                     "flowering_doy_median": q("flowering_doy", .5), "flowering_minus_sos_median": q("flowering_minus_sos", .5),
                     "maturity_doy_median": q("maturity_doy", .5), "maturity_minus_eos_p10": q("maturity_minus_eos", .1),
                     "maturity_minus_eos_median": q("maturity_minus_eos", .5),
                     "maturity_minus_eos_p90": q("maturity_minus_eos", .9)})
    return pd.DataFrame(rows)


def plot(summary: pd.DataFrame, path: str) -> None:
    order = (summary[summary["run"] == "new"].sort_values("seasons", ascending=False)["country"].tolist()
             + sorted(set(summary["country"]) - set(summary.loc[summary["run"] == "new", "country"])))
    panels = [("yield_p50", "yield median (t/ha)\nline = p10-p90", ("yield_p10", "yield_p90")),
              ("near_zero_share", f"share of seasons\n< {NEAR_ZERO_T_HA} t/ha", None),
              ("matured_by_eos_share", "share matured\nby calendar EOS", None),
              ("flowering_minus_sos_median", "flowering − SOS\n(days, median)", None),
              ("maturity_minus_eos_median", "maturity − EOS (days)\nmedian, line = p10-p90",
               ("maturity_minus_eos_p10", "maturity_minus_eos_p90"))]
    fig, axes = plt.subplots(1, len(panels), figsize=(14, 0.32 * len(order) + 1.6), sharey=True)
    y = {c: i for i, c in enumerate(order)}
    for ax, (col, title, span) in zip(axes, panels):
        for k, (run, g) in enumerate(summary.groupby("run")):
            pos = [y[c] + (0.18 if run == "new" else -0.18) for c in g["country"]]
            if span:
                ax.hlines(pos, g[span[0]], g[span[1]], color=COLORS[run], lw=1.2)
            ax.scatter(g[col], pos, s=22, color=COLORS[run], label=run, zorder=3)
        ax.set_title(title, loc="left", fontsize=9)
        ax.grid(axis="x", color="#e6e5e0")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[3].axvline(0, color="#52514e", lw=0.8)
    axes[4].axvline(0, color="#52514e", lw=0.8)
    axes[0].set_yticks(range(len(order)), order)
    axes[0].set_ylim(len(order) - 0.5, -0.5)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("WOFOST season sanity per country (grey = old FR/BE/DK run, blue = EU grid run)",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Per-country sanity check of WOFOST runs.")
    p.add_argument("--run", nargs=3, action="append", metavar=("NAME", "MANIFEST", "LOCATIONS_CSV"), required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    seasons = pd.concat([load_run(n, m, l, args.workers) for n, m, l in args.run], ignore_index=True)
    seasons.drop(columns=["error", "daily_path", "summary_path"], errors="ignore").to_csv(
        os.path.join(args.out, "season_dates.csv.gz"), index=False)
    summary = summarize(seasons)
    summary.to_csv(os.path.join(args.out, "sanity_per_country.csv"), index=False, float_format="%.3f")
    pd.set_option("display.width", 250)
    print(summary.to_string(index=False, float_format="%.2f"))
    plot(summary, os.path.join(args.out, "sanity_per_country.png"))


if __name__ == "__main__":
    main()
