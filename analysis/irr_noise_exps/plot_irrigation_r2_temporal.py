"""Quick plot: the irrigation-map comparison in R2 and temporal r (seed 0), same runs, colours and layout as `figures/irrigation_map_seed0.png`.

    uv run python -m analysis.irr_noise_exps.plot_irrigation_r2_temporal

The benchmark's per_run.csv already has r2 and temporal_r, so nothing is rerun. Writes `figures/irrigation_map_seed0_r2_temporal_r.png` and
`irrigation_map_r2_temporal_r.csv` (mean over the 18 countries) here.
"""

import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis.irr_noise_exps.plot_separate import BASE, W  # noqa: E402

BENCH = "/projects/prjs1788/tnp-crop/cybench_output"
OUT = os.path.dirname(os.path.abspath(__file__))
IRRIGATION = ("exp_3_irrigation_map", W, 0.0, "#2a78d6")
RUNS = {**BASE, "+ irrigation map (MIRCA-OS 2015)": IRRIGATION}
METRICS = {"r2": "R² (higher is better)", "temporal_r": "temporal r (higher is better)"}


def load(exp, setting, seed):
    d = pd.read_csv(f"{BENCH}/{exp}/per_run.csv")
    d = d[d.setting == setting]
    if seed is not None:
        d = d[d.seed == seed]
    assert len(d) == 18, (exp, setting, len(d))
    return d.set_index("country")[["r2", "temporal_r"]]


def main():
    data = {label: load(*spec[:3]) for label, spec in RUNS.items()}
    countries = sorted(next(iter(data.values())).index)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 1, figsize=(16, 7.8), sharex=True)
    n = len(RUNS)
    w = 0.86 / n
    x = np.arange(len(countries) + 1) + (np.arange(len(countries) + 1) >= len(countries)) * 0.6  # a gap before the mean group
    for ax, (metric, ylabel) in zip(axes, METRICS.items()):
        for i, (label, spec) in enumerate(RUNS.items()):
            vals = data[label].loc[countries, metric].tolist() + [data[label][metric].mean()]
            pos = x + (i - (n - 1) / 2) * w
            ax.bar(pos, vals, w, color=spec[3], label=label)
            ax.text(pos[-1], vals[-1] + (0.01 if vals[-1] >= 0 else -0.025), f"{vals[-1]:.3f}", ha="center", fontsize=6, rotation=90)
        ax.axvline(len(countries) - 0.3, color="k", lw=0.7, ls=":")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#e5e5e5", lw=0.6)
        ax.set_axisbelow(True)
    axes[1].set_xticks(x, countries + ["mean of all"])
    axes[0].legend(fontsize=8, ncol=3, frameon=False, loc="upper left")
    fig.suptitle("CY-Bench walk-forward, 18 European maize countries: irrigation map, R² and temporal r (pretraining seed 0)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/figures/irrigation_map_seed0_r2_temporal_r.png", dpi=150)
    summary = pd.DataFrame({label: {m: d[m].mean() for m in METRICS} for label, d in data.items()}).T.round(4)
    summary.to_csv(f"{OUT}/irrigation_map_r2_temporal_r.csv")
    print(summary.to_string())


if __name__ == "__main__":
    main()
