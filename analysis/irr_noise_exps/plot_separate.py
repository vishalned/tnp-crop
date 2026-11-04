"""The irrigation-map run and the observational-noise run, each in its own figure (same names and colours as the other clean bar plots).

    uv run python -m analysis.irr_noise_exps.plot_separate

CY-Bench walk-forward, 18 European maize countries, single pass, pretraining seed 0 for every pretrained run (the default TabICL is one run).
Both new runs use log-uniform points per table, so each figure ends with "+ log-uniform points per table" and the run being tested.
Writes `figures/irrigation_map_seed0.png`, `figures/observational_noise_seed0.png` and `separate_seed0.csv` (mean over countries).
"""

import os

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BENCH = "/projects/prjs1788/tnp-crop/cybench_output"
OUT = os.path.dirname(os.path.abspath(__file__))
W = "tabicl_wofost:wofost"
# label -> (experiment folder, setting, seed or None, colour); every file is that experiment's per_run.csv
BASE = {
    "TabICL default": ("exp_1_eu_grid_dense_omfix", "tabicl:tabicl_default_single", None, "#7a7a7a"),
    "SCM control": ("exp_1_eu_grid", "tabicl_wofost:scm", 0.0, "#f2c9a0"),
    "WOFOST, EU grid": ("exp_1_eu_grid", W, 0.0, "#8f7cc4"),
    "WOFOST, densified (old soil)": ("exp_1_eu_grid_dense", W, 0.0, "#7fcdb5"),
    "WOFOST, densified (OM fix, new soil)": ("exp_1_eu_grid_dense_omfix", W, 0.0, "#1b9e77"),
    "+ log-uniform points per table": ("exp_2a_loguniform_points", W, 0.0, "#e08a1e"),
}
FIGURES = {
    "irrigation_map_seed0": ("+ irrigation map (MIRCA-OS 2015)", ("exp_3_irrigation_map", W, 0.0, "#2a78d6"), "irrigation map"),
    "observational_noise_seed0": ("+ observational noise (fPAR, ssm, rsm)", ("exp_4_obs_noise", W, 0.0, "#b3122e"), "observational noise"),
}


def load(exp, setting, seed):
    d = pd.read_csv(f"{BENCH}/{exp}/per_run.csv")
    d = d[d.setting == setting]
    if seed is not None:
        d = d[d.seed == seed]
    assert len(d) == 18, (exp, setting, len(d))
    return d.set_index("country")[["nrmse", "anomaly_r"]]


def draw(runs, path, title):
    data = {label: load(*spec[:3]) for label, spec in runs.items()}
    countries = sorted(next(iter(data.values())).index)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 1, figsize=(16, 7.8), sharex=True)
    n = len(runs)
    w = 0.86 / n
    x = np.arange(len(countries) + 1) + (np.arange(len(countries) + 1) >= len(countries)) * 0.6  # gap before the mean group
    for ax, metric, ylabel in ((axes[0], "nrmse", "NRMSE (lower is better)"), (axes[1], "anomaly_r", "anomaly r (higher is better)")):
        for i, (label, spec) in enumerate(runs.items()):
            vals = data[label].loc[countries, metric].tolist() + [data[label][metric].mean()]
            pos = x + (i - (n - 1) / 2) * w
            ax.bar(pos, vals, w, color=spec[3], label=label)
            ax.text(pos[-1], vals[-1] + (0.003 if metric == "nrmse" else 0.01), f"{vals[-1]:.3f}", ha="center", fontsize=6, rotation=90)
        ax.axvline(len(countries) - 0.3, color="k", lw=0.7, ls=":")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#e5e5e5", lw=0.6)
        ax.set_axisbelow(True)
    axes[1].set_xticks(x, countries + ["mean of all"])
    axes[0].legend(fontsize=8, ncol=3, frameon=False, loc="upper left")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {label: {"nrmse_mean": d.nrmse.mean(), "anomaly_r_mean": d.anomaly_r.mean()} for label, d in data.items()}


def main():
    rows = []
    for name, (label, spec, short) in FIGURES.items():
        res = draw({**BASE, label: spec}, f"{OUT}/figures/{name}.png",
                   f"CY-Bench walk-forward, 18 European maize countries: {short} (pretraining seed 0)")
        rows += [{"figure": name, "run": r, **v} for r, v in res.items()]
    out = pd.DataFrame(rows)
    out.round(4).to_csv(f"{OUT}/separate_seed0.csv", index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
