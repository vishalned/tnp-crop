"""Sanity test: does the benchmark gain come from the whole of Europe's WOFOST data or from one part of it?

    uv run python -m analysis.sanity_regions.plot_overview

Pretraining (exp_1 recipe, 2000 steps, seed 0) on the densified EU grid table (old soil, uniform points per table) restricted to a regional
subset of the countries (`scripts/slurm/submit_sanity_regions.sh`), benchmarked on all 18 European CY-Bench countries (walk-forward, single pass).
Compared with default TabICL and with pretraining on all 19 countries of the same table (seed 0 of exp_1_eu_grid_dense). Writes `figures/overview.png`
and `overview.csv`; right of the dotted line: mean over countries and mean weighted by region-years.
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
    ("Pretrained on all 19 countries", "exp_1_eu_grid_dense", W, 0.0, "per_run.csv", "#1b9e77"),
    ("north-west only: FR BE NL DE DK", "sanity_regions_nw", W, 0.0, "per_run.csv", "#a6761d"),
    ("central-east only: PL CZ SK HU RO BG AT HR SI LT", "sanity_regions_ce", W, 0.0, "per_run.csv", "#e6ab02"),
    ("south only: ES PT IT EL", "sanity_regions_south", W, 0.0, "per_run.csv", "#e7298a"),
    ("FR BE DK only", "sanity_regions_frbedk", W, 0.0, "per_run.csv", "#17becf"),
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
    overall.round(4).to_csv(f"{OUT}/overview.csv")

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
    fig.suptitle("CY-Bench walk-forward, 18 European maize countries: WOFOST pretraining on regional subsets of the densified table (old soil), seed 0", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figures/overview.png", dpi=150)
    print(overall.round(4).to_string())


if __name__ == "__main__":
    main()
