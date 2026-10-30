"""Collect the partial results, write the long CSV, the aggregates, the figures and the summary.

    poetry run python -m analysis.feature_importance.aggregate_and_plot
Outputs in analysis/feature_importance/results/.
"""

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PARTS = "/projects/prjs1788/tnp-crop/feature_importance/parts"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
MODELS = ["default_tabicl", "scm", "wofost"]
LABEL = {"default_tabicl": "Default TabICL", "scm": "SCM control", "wofost": "WOFOST (dense)"}
COLOR = {"default_tabicl": "#7a7a7a", "scm": "#d95f02", "wofost": "#1b9e77"}


def load() -> pd.DataFrame:
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{PARTS}/*.csv"))], ignore_index=True)
    df.to_csv(f"{OUT}/importance_long.csv", index=False)
    return df


def per_unit(df: pd.DataFrame) -> pd.DataFrame:
    """Mean over the permutation seeds per (model, data, unit, group)."""
    keys = ["model", "data", "unit", "country", "group", "group_view"]
    return df.groupby(keys, as_index=False).agg(delta_nrmse=("delta_nrmse", "mean"), delta_anomaly_r=("delta_anomaly_r", "mean"),
                                                n_rows=("n_rows", "first"))


def aggregate(units: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, data, group, view), g in units.groupby(["model", "data", "group", "group_view"]):
        w = g["n_rows"].to_numpy(float)
        rows.append({"model": model, "data": data, "group": group, "group_view": view, "n_units": len(g),
                     "delta_nrmse_mean": g["delta_nrmse"].mean(), "delta_nrmse_sd": g["delta_nrmse"].std(ddof=1),
                     "delta_nrmse_weighted": np.average(g["delta_nrmse"], weights=w),
                     "delta_anomaly_r_mean": g["delta_anomaly_r"].mean(), "delta_anomaly_r_weighted":
                         np.average(g["delta_anomaly_r"], weights=w) if g["delta_anomaly_r"].notna().all() else np.nan})
    agg = pd.DataFrame(rows)
    agg.to_csv(f"{OUT}/importance_aggregated.csv", index=False)
    return agg


def bars(agg: pd.DataFrame, data: str, view: str, title: str, path: str) -> None:
    a = agg[(agg.data == data) & (agg.group_view == view)]
    order = a[a.model == "wofost"].sort_values("delta_nrmse_mean", ascending=False)["group"].tolist()
    x, w = np.arange(len(order)), 0.27
    fig, ax = plt.subplots(figsize=(max(7, 0.7 * len(order) + 2), 4.2))
    for i, m in enumerate(MODELS):
        s = a[a.model == m].set_index("group").reindex(order)
        ax.bar(x + (i - 1) * w, s["delta_nrmse_mean"], w, yerr=s["delta_nrmse_sd"], color=COLOR[m], label=LABEL[m],
               capsize=2, error_kw={"lw": 0.8})
    ax.set_xticks(x, order, rotation=45, ha="right")
    ax.set_ylabel("Δ NRMSE (permuted − baseline)")
    ax.set_title(title)
    ax.axhline(0, color="k", lw=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def shortcut_bars(agg: pd.DataFrame, view: str, path: str) -> None:
    a = agg[(agg.model == "wofost") & (agg.group_view == view)]
    piv = a.pivot(index="group", columns="data", values="delta_nrmse_mean")
    sd = a.pivot(index="group", columns="data", values="delta_nrmse_sd")
    order = piv.sort_values("simulated", ascending=False).index
    x, w = np.arange(len(order)), 0.38
    fig, ax = plt.subplots(figsize=(max(7, 0.7 * len(order) + 2), 4.2))
    ax.bar(x - w / 2, piv.loc[order, "simulated"], w, yerr=sd.loc[order, "simulated"], color="#1b9e77", label="simulated tables", capsize=2, error_kw={"lw": 0.8})
    ax.bar(x + w / 2, piv.loc[order, "cybench"], w, yerr=sd.loc[order, "cybench"], color="#7570b3", label="CY-Bench", capsize=2, error_kw={"lw": 0.8})
    ax.set_xticks(x, order, rotation=45, ha="right")
    ax.set_ylabel("Δ NRMSE (permuted − baseline)")
    ax.set_title(f"WOFOST model: simulated vs CY-Bench ({view} groups)")
    ax.axhline(0, color="k", lw=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def heatmap(units: pd.DataFrame, path: str) -> None:
    u = units[units.data == "cybench"]
    order = (u[(u.model == "wofost") & (u.group_view == "single")].groupby("group")["delta_nrmse"].mean()
             .sort_values(ascending=False).index.tolist())
    order += sorted(u[u.group_view == "family"]["group"].unique())
    countries = ["DE", "FR", "IT", "ES", "RO", "HU", "PL"]
    vmax = np.nanpercentile(np.abs(u["delta_nrmse"]), 98)
    fig, axes = plt.subplots(1, 3, figsize=(17, 3.8), sharey=True)
    for ax, m in zip(axes, MODELS):
        piv = u[u.model == m].pivot(index="country", columns="group", values="delta_nrmse").reindex(index=countries, columns=order)
        im = ax.imshow(piv.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(order)), order, rotation=70, ha="right", fontsize=7)
        ax.set_yticks(range(len(countries)), countries)
        ax.set_title(LABEL[m])
        ax.axvline(len([g for g in order if not g.startswith("fam_")]) - 0.5, color="k", lw=0.8)
    fig.colorbar(im, ax=axes, label="Δ NRMSE", shrink=0.9)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def summary(agg: pd.DataFrame, df: pd.DataFrame) -> None:
    lines = ["# Grouped permutation importance: summary", "",
             "Δ NRMSE = NRMSE(permuted) − NRMSE(baseline), mean over 3 permutation seeds; CY-Bench: mean over the 7 countries "
             "(DE FR IT ES RO HU PL), simulated: mean over tables. Single-variable groups only unless stated.", ""]
    single = agg[agg.group_view == "single"]
    for data, name in (("cybench", "CY-Bench"), ("simulated", "Simulated tables")):
        lines += [f"## Top 3 groups, {name}", ""]
        for m in MODELS:
            s = single[(single.model == m) & (single.data == data)].sort_values("delta_nrmse_mean", ascending=False).head(3)
            lines.append(f"- **{LABEL[m]}**: " + ", ".join(f"{r.group} ({r.delta_nrmse_mean:+.4f})" for r in s.itertuples()))
        lines.append("")
    lines += ["## Simulated importance far above CY-Bench (>3x and >0.01 absolute), single groups", ""]
    for m in MODELS:
        p = single[single.model == m].pivot(index="group", columns="data", values="delta_nrmse_mean")
        flag = p[(p["simulated"] > 3 * p["cybench"].clip(lower=1e-4)) & (p["simulated"] - p["cybench"] > 0.01)]
        lines.append(f"- **{LABEL[m]}**: " + (", ".join(f"{g} (sim {r.simulated:+.4f} vs CY-Bench {r.cybench:+.4f})" for g, r in flag.iterrows()) or "none"))
    lines += ["", "## Share of the total importance in the biggest group (sum of positive mean Δ NRMSE, single groups)", ""]
    for data, name in (("cybench", "CY-Bench"), ("simulated", "Simulated")):
        for m in MODELS:
            s = single[(single.model == m) & (single.data == data)]
            pos = s["delta_nrmse_mean"].clip(lower=0)
            top = s.loc[pos.idxmax()]
            lines.append(f"- {name}, {LABEL[m]}: {top.group} = {pos.max() / pos.sum():.0%} of the total")
    lines += ["", "## CY-Bench: all groups (mean over countries / weighted by region-years, Δ anomaly r)", ""]
    t = agg[(agg.data == "cybench")].pivot(index=["group_view", "group"], columns="model",
                                            values=["delta_nrmse_mean", "delta_nrmse_weighted", "delta_anomaly_r_mean"])
    lines += ["```", t.round(4).to_string(), "```", ""]
    checks = {}
    for f in glob.glob(f"{PARTS}/*_check.json"):
        d = json.load(open(f))["max_abs_diff_per_year"]
        checks[os.path.basename(f).replace("_check.json", "")] = max(d.values())
    lines += ["## Baseline vs the existing exp_1 predictions (max |diff| in t/ha over folds)", ""]
    lines += [f"- {k}: {v:.4g}" for k, v in sorted(checks.items())]
    open(f"{OUT}/summary.md", "w").write("\n".join(lines) + "\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    df = load()
    units = per_unit(df)
    agg = aggregate(units)
    for view in ("single", "family"):
        bars(agg, "cybench", view, f"Δ NRMSE per {view} group, CY-Bench (mean ± sd over countries)", f"{OUT}/bars_cybench_{view}.png")
        bars(agg, "simulated", view, f"Δ NRMSE per {view} group, simulated tables (mean ± sd over tables)", f"{OUT}/bars_simulated_{view}.png")
        shortcut_bars(agg, view, f"{OUT}/bars_wofost_sim_vs_cybench_{view}.png")
    heatmap(units, f"{OUT}/heatmap_countries_groups.png")
    summary(agg, df)
    print("written to", OUT)


if __name__ == "__main__":
    main()
