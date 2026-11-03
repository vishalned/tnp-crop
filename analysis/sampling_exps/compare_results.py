"""CY-Bench results of the two table-sampling experiments against the earlier runs.

    uv run python -m analysis.sampling_exps.compare_results

Experiment A (`exp_2a_loguniform_points`): points per crop table log-uniform from 1 up to the cap (was uniform from 10).
Experiment B (`exp_2b_loguniform_points_scm`): A, plus an SCM (prior) table in 50 % of the steps, with table sizes drawn
from the same sampler. Both train on the densified OM-fix table, 3 seeds, same recipe as exp_1. Each run's mean over
seeds per country comes from its `summary.csv` (18 European countries, walk-forward, single pass).

Writes `comparison_per_country.csv`, `comparison_overall.csv` and `figures/comparison.png` (two rows of per-country bars:
NRMSE and anomaly r) here.
"""

import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BENCH = "/projects/prjs1788/tnp-crop/cybench_output"
OUT = os.path.dirname(os.path.abspath(__file__))
W = "tabicl_wofost:wofost"
# label -> (experiment folder, setting in its summary.csv)
RUNS = {
    "TabICL default": ("exp_1_eu_grid_dense_omfix", "tabicl:tabicl_default_single"),
    "SCM control (EU grid shapes)": ("exp_1_eu_grid", "tabicl_wofost:scm"),
    "WOFOST, exp_1 points": ("exp_1", W),
    "WOFOST, EU grid": ("exp_1_eu_grid", W),
    "WOFOST, densified, old soil": ("exp_1_eu_grid_dense", W),
    "WOFOST, densified, OM fix (baseline)": ("exp_1_eu_grid_dense_omfix", W),
    "A: log-uniform points": ("exp_2a_loguniform_points", W),
    "B: log-uniform points + 50 % SCM": ("exp_2b_loguniform_points_scm", "tabicl_wofost:mix"),
}
# the clean figure (same names and colours as analysis/soc_fix): label in the figure -> (run above, colour)
PLOT = {
    "TabICL default": ("TabICL default", "#7a7a7a"),
    "SCM control": ("SCM control (EU grid shapes)", "#f2c9a0"),
    "WOFOST, EU grid": ("WOFOST, EU grid", "#8f7cc4"),
    "WOFOST, densified (old soil)": ("WOFOST, densified, old soil", "#7fcdb5"),
    "WOFOST, densified (OM fix, new soil)": ("WOFOST, densified, OM fix (baseline)", "#1b9e77"),
    "+ log-uniform points per table": ("A: log-uniform points", "#e08a1e"),
    "+ log-uniform points + 50 % SCM tables": ("B: log-uniform points + 50 % SCM", "#b3122e"),
}
BASELINE, A, B = list(RUNS)[5], list(RUNS)[6], list(RUNS)[7]


def draw(per_country, countries, plot, path, suptitle):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 1, figsize=(16, 7.8), sharex=True)
    n = len(plot)
    w = 0.86 / n
    x = np.arange(len(countries) + 1) + (np.arange(len(countries) + 1) >= len(countries)) * 0.6  # gap before the mean group
    for ax, metric, title in ((axes[0], "nrmse", "NRMSE (lower is better)"), (axes[1], "anomaly_r", "anomaly r (higher is better)")):
        for i, (label, (run, color)) in enumerate(plot.items()):
            vals = per_country[metric][run].tolist() + [per_country[metric][run].mean()]
            pos = x + (i - (n - 1) / 2) * w
            ax.bar(pos, vals, w, color=color, label=label)
            ax.text(pos[-1], vals[-1] + (0.003 if metric == "nrmse" else 0.01), f"{vals[-1]:.3f}", ha="center", fontsize=6, rotation=90)
        ax.axvline(len(countries) - 0.3, color="k", lw=0.7, ls=":")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylabel(title)
        ax.grid(axis="y", color="#e5e5e5", lw=0.6)
        ax.set_axisbelow(True)
    axes[1].set_xticks(x, countries + ["mean of all"])
    axes[0].legend(fontsize=8, ncol=3, frameon=False, loc="upper left")
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(path, dpi=150)

def main():
    frames = []
    for label, (exp, setting) in RUNS.items():
        path = f"{BENCH}/{exp}/summary.csv"
        if not os.path.exists(path):
            print(f"missing: {label} ({path})")
            continue
        s = pd.read_csv(path)
        s = s[s.setting == setting].assign(run=label)
        frames.append(s[["run", "country", "nrmse", "nrmse_sd", "anomaly_r", "anomaly_r_sd", "r2", "n_region_years", "n_seeds"]])
    df = pd.concat(frames, ignore_index=True)
    runs = [r for r in RUNS if r in set(df.run)]
    per_country = df.pivot(index="country", columns="run", values=["nrmse", "anomaly_r"]).reindex(columns=runs, level=1)
    per_country.round(4).to_csv(f"{OUT}/comparison_per_country.csv")

    rows = []
    for run in runs:
        g = df[df.run == run]
        w = g.n_region_years
        rows.append({"run": run, "n_countries": len(g), "n_seeds": int(g.n_seeds.min()), "nrmse_mean": g.nrmse.mean(),
                     "nrmse_weighted": np.average(g.nrmse, weights=w), "anomaly_r_mean": g.anomaly_r.mean(),
                     "anomaly_r_weighted": np.average(g.anomaly_r, weights=w)})
    overall = pd.DataFrame(rows).set_index("run")
    overall.round(4).to_csv(f"{OUT}/comparison_overall.csv")
    print(overall.round(4).to_string())
    for new in (A, B):
        if new in runs:
            d = per_country["nrmse"][new] - per_country["nrmse"][BASELINE]
            a = per_country["anomaly_r"][new] - per_country["anomaly_r"][BASELINE]
            print(f"{new} vs OM-fix baseline: NRMSE better in {(d < 0).sum()} of {len(d)} countries "
                  f"(mean diff {d.mean():+.4f}); anomaly r higher in {(a > 0).sum()} (mean diff {a.mean():+.4f})")

    countries = per_country.index.tolist()
    title = "CY-Bench walk-forward, 18 European maize countries: table-sampling experiments (mean over 3 pretraining seeds)"
    draw(per_country, countries, PLOT, f"{OUT}/figures/comparison_with_50p_scm.png", title)
    no_scm = {k: v for k, v in PLOT.items() if "SCM tables" not in k}  # without the 50 % SCM run (B)
    draw(per_country, countries, no_scm, f"{OUT}/figures/comparison.png", title)

if __name__ == "__main__":
    main()
