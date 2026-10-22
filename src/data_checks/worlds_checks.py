"""Checks of the sampled-worlds datasets (A, A_w, B, C, D, ...; `generate_worlds_dataset.py`).

    uv run python -m src.data_checks.worlds_checks --root /projects/prjs1788/tnp-crop/wofost_worlds_test \
        --datasets A A_w B C D --reference-manifest /projects/prjs1788/tnp-crop/wofost_calendar/dataset_manifest.csv \
        --locations-csv data/raw/locations/locations_maize.csv --out <dir>

Reads `<root>/<dataset>/{dataset_manifest.csv,worlds.parquet}` and, when present,
`<root>/<dataset>/distribution_gap/gap_stats.csv` (`nanotabicl.distribution_gap` on that dataset's
processed table). Writes to `--out` and prints:

1. `check1_reproduction.csv`: dataset A against the reference manifest (same location, year,
   jitter): sowing dates and yields;
2. `check2_timing.csv` / `check2_timing.png`: flowering and maturity day of year per dataset and
   country, against the crop calendar's SOS/EOS;
3. `check3_yield.csv` / `check3_yield.png`: yield per dataset and country (kept worlds);
4. `check4_correlation.csv`: agreement of the feature-yield correlations with CY-Bench (share of
   features with the same sign, correlation of the correlations), per dataset and comparison;
5. `check5_runtime.csv`: runs, rejected worlds and runtime per dataset, and the CPU time for
   100k / 300k runs at the measured rate.
"""

import argparse
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data_pipeline.wofost.worlds import season_end  # noqa: E402

COLORS = ["#8a8984", "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e34948"]


def load(root: str, datasets: list, locations_csv: str) -> pd.DataFrame:
    countries = pd.read_csv(locations_csv)["country"]
    frames = []
    for name in datasets:
        m = pd.read_csv(os.path.join(root, name, "dataset_manifest.csv"))
        m["dataset"] = name
        m["country"] = m["location_index"].map(countries)
        frames.append(m)
    return pd.concat(frames, ignore_index=True)


def doy(dates: pd.Series) -> pd.Series:
    return pd.to_datetime(dates, errors="coerce").dt.dayofyear


def check1(manifests: pd.DataFrame, reference_path: str) -> pd.DataFrame:
    a = manifests[manifests["dataset"] == "A"]
    ref = pd.read_csv(reference_path)
    keys = ["location_index", "year", "jitter_index"]
    both = a.merge(ref, on=keys, suffixes=("", "_ref"))
    diff = (both["yield_kg_per_ha"] - both["yield_kg_per_ha_ref"]).abs()
    return pd.DataFrame([{
        "runs_A": len(a), "matched_in_reference": len(both),
        "same_sowing_date": int((both["sowing_date"] == both["sowing_date_ref"]).sum()),
        "max_abs_yield_diff_kg_ha": float(diff.max()), "mean_abs_yield_diff_kg_ha": float(diff.mean()),
        "identical_yields": int((diff < 1e-6).sum()),
    }])


def kept(manifests: pd.DataFrame) -> pd.DataFrame:
    ok = manifests["status"] == "success"
    if "world_kept" in manifests:
        ok &= manifests["world_kept"].astype(bool)
    return manifests[ok]


def check2(manifests: pd.DataFrame) -> pd.DataFrame:
    m = kept(manifests).assign(flowering_doy=lambda d: doy(d["flowering_date"]),
                               maturity_doy=lambda d: doy(d["maturity_date"]))
    rows = []
    for (ds, country), g in m.groupby(["dataset", "country"]):
        eos = [season_end(int(y), int(s), int(e)) for y, s, e in zip(g["year"], g["sos_doy"], g["eos_doy"])]
        mat = pd.to_datetime(g["maturity_date"], errors="coerce").dt.date
        rows.append({"dataset": ds, "country": country, "runs": len(g),
                     "sos_doy_median": g["sos_doy"].median(), "eos_doy_median": g["eos_doy"].median(),
                     **{f"flowering_doy_{q}": g["flowering_doy"].quantile(p) for q, p in (("p10", .1), ("p50", .5), ("p90", .9))},
                     **{f"maturity_doy_{q}": g["maturity_doy"].quantile(p) for q, p in (("p10", .1), ("p50", .5), ("p90", .9))},
                     "matured_by_eos": float(np.mean([pd.notna(d) and d <= e for d, e in zip(mat, eos)]))})
    return pd.DataFrame(rows)


def check3(manifests: pd.DataFrame) -> pd.DataFrame:
    m = kept(manifests).assign(yield_t_ha=lambda d: d["yield_kg_per_ha"] / 1000)
    g = m.groupby(["dataset", "country"])["yield_t_ha"]
    return pd.concat([g.size().rename("runs"), g.mean().rename("mean"), g.std().rename("std"),
                      g.quantile(.1).rename("p10"), g.median().rename("p50"), g.quantile(.9).rename("p90")],
                     axis=1).reset_index()


def check4(root: str, datasets: list) -> pd.DataFrame:
    rows = []
    for ds in datasets:
        path = os.path.join(root, ds, "distribution_gap", "gap_stats.csv")
        if not os.path.exists(path):
            continue
        st = pd.read_csv(path)
        for comparison, g in st[st["feature"] != "yield"].groupby("comparison"):
            both = g[["corr_yield_wofost", "corr_yield_cybench"]].dropna()
            rows.append({"dataset": ds, "comparison": comparison, "features": len(both),
                         "same_sign": float(np.mean(np.sign(both.iloc[:, 0]) == np.sign(both.iloc[:, 1]))),
                         "r_of_correlations": float(both.corr().iloc[0, 1])})
    return pd.DataFrame(rows)


def check5(root: str, manifests: pd.DataFrame, datasets: list) -> pd.DataFrame:
    rows = []
    for ds in datasets:
        m = manifests[manifests["dataset"] == ds]
        worlds_path = os.path.join(root, ds, "worlds.parquet")
        worlds = pd.read_parquet(worlds_path) if os.path.exists(worlds_path) else None
        per_run = m["runtime_s"].mean()
        pp = int((m.get("production", pd.Series(dtype=str)) == "PP").sum())
        rows.append({"dataset": ds, "locations": m["location_index"].nunique(),
                     "worlds": len(worlds) if worlds is not None else None,
                     "worlds_rejected": int((~worlds["kept"]).sum()) if worlds is not None else None,
                     "runs": len(m), "runs_failed": int((m["status"] != "success").sum()), "runs_PP": pp,
                     "cpu_s_total": float(m["runtime_s"].sum()), "cpu_s_per_run": float(per_run),
                     "cpu_h_100k_runs": float(per_run * 1e5 / 3600), "cpu_h_300k_runs": float(per_run * 3e5 / 3600)})
    return pd.DataFrame(rows)


def _boxes(ax, groups: list, labels: list, colors: list) -> None:
    b = ax.boxplot(groups, vert=False, patch_artist=True, widths=0.6, showfliers=False,
                   medianprops={"color": "white", "lw": 1.5})
    for patch, c in zip(b["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_edgecolor(c)
    ax.set_yticks(range(1, len(labels) + 1), labels)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plots(manifests: pd.DataFrame, datasets: list, out: str) -> None:
    m = kept(manifests)
    countries = sorted(m["country"].dropna().unique())
    color = dict(zip(datasets, COLORS))
    labels = [f"{c} · {ds}" for c in countries for ds in datasets]

    def groups(values):
        return [values[(m["country"] == c) & (m["dataset"] == ds)].dropna().to_numpy() for c in countries for ds in datasets]

    fig, axes = plt.subplots(1, 2, figsize=(10, 0.32 * len(labels) + 1.2), sharey=True)
    for ax, (col, title) in zip(axes, (("flowering_date", "flowering"), ("maturity_date", "maturity"))):
        g = groups(doy(m[col]))
        _boxes(ax, [x if len(x) else np.array([np.nan]) for x in g], labels, [color[ds] for _ in countries for ds in datasets])
        for i, c in enumerate(countries):
            cal = m[m["country"] == c]
            ax.axvline(cal["eos_doy"].median(), color="#52514e", ls=":", lw=1)
        ax.set_title(f"{title} (day of year); dotted = calendar EOS", loc="left", fontsize=10)
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "check2_timing.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 0.32 * len(labels) + 1.2))
    _boxes(ax, [x if len(x) else np.array([np.nan]) for x in groups(m["yield_kg_per_ha"] / 1000)], labels,
           [color[ds] for _ in countries for ds in datasets])
    ax.invert_yaxis()
    ax.set_xlabel("simulated yield (t/ha, kept worlds)")
    ax.set_title("yield per dataset and country (box = p25-p75, whiskers 1.5 IQR)", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "check3_yield.png"), dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Checks of the sampled-worlds datasets.")
    p.add_argument("--root", required=True)
    p.add_argument("--datasets", nargs="+", default=["A", "A_w", "B", "C", "D"])
    p.add_argument("--reference-manifest", default=None, help="Manifest dataset A should reproduce (check 1).")
    p.add_argument("--locations-csv", default="data/raw/locations/locations_maize.csv")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)

    manifests = load(args.root, args.datasets, args.locations_csv)
    results = {}
    if args.reference_manifest and "A" in args.datasets:
        results["check1_reproduction"] = check1(manifests, args.reference_manifest)
    results["check2_timing"] = check2(manifests)
    results["check3_yield"] = check3(manifests)
    results["check4_correlation"] = check4(args.root, args.datasets)
    results["check5_runtime"] = check5(args.root, manifests, args.datasets)
    for name, df in results.items():
        df.to_csv(os.path.join(args.out, f"{name}.csv"), index=False)
        print(f"\n== {name}\n{df.to_string(index=False, float_format='%.3f')}")
    for ds in args.datasets:
        path = os.path.join(args.root, ds, "world_rejections.csv")
        if os.path.exists(path):
            print(f"\n== rejections {ds}\n{pd.read_csv(path).to_string(index=False, float_format='%.3f')}")
    plots(manifests, args.datasets, args.out)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
