"""Per-country bar plots of CY-Bench results, drawn from `results/runs.csv` (see `analysis/collect_results.py`).

A plot is a list of runs (`analysis/runs.yaml` has the named ones, with their legend label and colour) and a few options; `analysis/make_plots.py`
draws every `experiments/*/plots.yaml`. Two rows of bars (NRMSE: lower is better; anomaly r: higher is better), the 18 countries and, right of a
dotted line, the mean over the countries ("mean of all") and, optionally, the mean weighted by region-years.
"""

import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS_CSV = os.path.join(os.path.dirname(HERE), "results", "runs.csv")


def load_runs(path: str = RUNS_CSV) -> pd.DataFrame:
    return pd.read_csv(path)


def values(runs: pd.DataFrame, experiment: str, setting: str, seed="mean") -> pd.DataFrame:
    """Per country nrmse, anomaly_r and n_region_years of one run: its mean over the pretraining seeds (seed="mean") or one seed.
    The default TabICL has no seed (one deterministic run) and is always taken whole."""
    d = runs[(runs.experiment == experiment) & (runs.setting == setting)]
    if seed != "mean" and d.seed.notna().any():
        d = d[d.seed == float(seed)]
    if d.empty:
        raise KeyError(f"no results for {experiment} / {setting} / seed {seed}")
    g = d.groupby("country")
    out = g[["nrmse", "anomaly_r"]].mean()
    out["n_region_years"] = g.n_region_years.first()
    if len(out) != 18:
        raise ValueError(f"{experiment} / {setting}: {len(out)} countries, expected 18")
    return out


def draw(entries: list, path: str, title: str, weighted: bool = False, figsize=(16, 7.8), title_size=None) -> pd.DataFrame:
    """entries: [{"label", "color", "values"}]. Returns the mean (and weighted mean) of each run."""
    countries = sorted(entries[0]["values"].index)
    summary = []
    for e in entries:
        v = e["values"].loc[countries]
        row = {"run": e["label"], "nrmse_mean": v.nrmse.mean(), "anomaly_r_mean": v.anomaly_r.mean()}
        if weighted:
            row.update(nrmse_weighted=np.average(v.nrmse, weights=v.n_region_years),
                       anomaly_r_weighted=np.average(v.anomaly_r, weights=v.n_region_years))
        summary.append(row)
    summary = pd.DataFrame(summary).set_index("run")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    n_groups = 2 if weighted else 1
    n, w = len(entries), 0.86 / len(entries)
    k = len(countries) + n_groups
    x = np.arange(k) + (np.arange(k) >= len(countries)) * 0.6  # a gap before the mean group(s)
    for ax, metric, ylabel in ((axes[0], "nrmse", "NRMSE (lower is better)"), (axes[1], "anomaly_r", "anomaly r (higher is better)")):
        for i, e in enumerate(entries):
            v = e["values"].loc[countries, metric].tolist() + [summary.loc[e["label"], f"{metric}_mean"]]
            if weighted:
                v.append(summary.loc[e["label"], f"{metric}_weighted"])
            pos = x + (i - (n - 1) / 2) * w
            ax.bar(pos, v, w, color=e["color"], label=e["label"])
            for p, val in zip(pos[-n_groups:], v[-n_groups:]):  # the numbers on the mean group(s)
                off = (0.004 if metric == "nrmse" else 0.01) if not weighted else (0.004 if metric == "nrmse" else 0.01) * (1 if val >= 0 else -2.5)
                ax.text(p, val + off, f"{val:.3f}", ha="center", fontsize=5.5 if weighted else 6, rotation=90)
        ax.axvline(len(countries) - 0.3, color="k", lw=0.7, ls=":")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#e5e5e5", lw=0.6)
        ax.set_axisbelow(True)
    axes[1].set_xticks(x, countries + (["mean", "weighted"] if weighted else ["mean of all"]))
    axes[0].legend(fontsize=8, ncol=3, frameon=False, loc="upper left")
    fig.suptitle(title, **({"fontsize": title_size} if title_size else {}))
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return summary
