"""Overview figure: the runs of this study side by side on CY-Bench, pretraining seed 0 only.

    uv run python -m analysis.irr_noise_exps.plot_seed0_overview

Per country (18 European maize countries, walk-forward, single pass) NRMSE and anomaly r of: default TabICL; WOFOST pretraining on the densified OM-fix table (seed 0);
and, on top of that table, log-uniform points per table, the MIRCA-OS irrigation map, and observational noise on fPAR / ssm / rsm (each seed 0).
Right of the dotted line: mean over the countries and mean weighted by region-years. Writes `figures/overview_seed0.png` and `overview_seed0.csv`.
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
# label, experiment folder, setting, seed (None: the file has one run), file, colour
RUNS = [
    ("TabICL default (single pass)", "exp_1_eu_grid_dense_omfix", "tabicl:tabicl_default_single", None, "per_run.csv", "#7a7a7a"),
    ("WOFOST pretraining, densified OM-fix table", "exp_1_eu_grid_dense_omfix", W, 0.0, "per_run.csv", "#1b9e77"),
    ("+ log-uniform points per table", "exp_2a_loguniform_points", W, 0.0, "per_run.csv", "#e08a1e"),
    ("+ irrigation map (MIRCA-OS 2015)", "exp_3_irrigation_map", W, 0.0, "per_run.csv", "#2a78d6"),
    ("+ observational noise (fPAR, ssm, rsm)", "exp_4_obs_noise", W, 0.0, "per_run.csv", "#b3122e"),
]


def main():
    frames = []
    for label, exp, setting, seed, file, _ in RUNS:
        d = pd.read_csv(f"{BENCH}/{exp}/{file}")
        d = d[d.setting == setting]
        if seed is not None:
            d = d[d.seed == seed]
        assert len(d) == 18, (label, len(d))
        frames.append(d.assign(run=label)[["run", "country", "nrmse", "anomaly_r", "n_region_years"]])
    df = pd.concat(frames, ignore_index=True)
    countries = sorted(df.country.unique())
    labels = [r[0] for r in RUNS]

    rows = []
    for label in labels:
        g = df[df.run == label].set_index("country")
        rows.append({"run": label, "nrmse_mean": g.nrmse.mean(), "nrmse_weighted": np.average(g.nrmse, weights=g.n_region_years),
                     "anomaly_r_mean": g.anomaly_r.mean(), "anomaly_r_weighted": np.average(g.anomaly_r, weights=g.n_region_years)})
    overall = pd.DataFrame(rows).set_index("run")
    overall.round(4).to_csv(f"{OUT}/overview_seed0.csv")

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 1, figsize=(17, 8.2), sharex=True)
    n = len(RUNS)
    w = 0.86 / n
    x = np.arange(len(countries) + 2) + (np.arange(len(countries) + 2) >= len(countries)) * 0.6  # a gap before the mean groups
    for ax, metric, title in ((axes[0], "nrmse", "NRMSE  (lower is better)"), (axes[1], "anomaly_r", "anomaly r  (higher is better)")):
        for i, (label, *_rest, color) in enumerate(RUNS):
            g = df[df.run == label].set_index("country")
            vals = [g.loc[c, metric] for c in countries] + [overall.loc[label, f"{metric}_mean"], overall.loc[label, f"{metric}_weighted"]]
            pos = x + (i - (n - 1) / 2) * w
            ax.bar(pos, vals, w, color=color, label=label)
            for p, v in zip(pos[-2:], vals[-2:]):  # numbers on the two mean groups
                ax.text(p, v + (0.004 if metric == "nrmse" else 0.01) * (1 if v >= 0 else -2.5), f"{v:.3f}", ha="center", fontsize=5.5, rotation=90)
        ax.axvline(len(countries) - 0.3, color="k", lw=0.7, ls=":")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylabel(title)
        ax.grid(axis="y", color="#e5e5e5", lw=0.6)
        ax.set_axisbelow(True)
    axes[1].set_xticks(x, countries + ["mean", "weighted"])
    axes[0].legend(fontsize=8, ncol=3, frameon=False, loc="upper left")
    fig.suptitle("CY-Bench walk-forward, 18 European maize countries: default TabICL and WOFOST pretraining variants (pretraining seed 0)", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figures/overview_seed0.png", dpi=150)
    print(overall.round(4).to_string())


if __name__ == "__main__":
    main()
