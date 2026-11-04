"""CY-Bench results of the irrigation-map run and the observational-noise run against the earlier runs.

    uv run python -m analysis.irr_noise_exps.compare_results

C (`exp_3_irrigation_map`): log-uniform points per table (run A's sampling), trained on the table where each location is irrigated (potential
production) or rainfed as the MIRCA-OS 2015 maize map says. D (`exp_4_obs_noise`): run A's setup on the rainfed OM-fix table, with observational
noise on fPAR / ssm / rsm in the training tables. Both: ONE pretraining seed (0). A's result is shown as its 3-seed mean and, for a fair pairing,
as its seed 0 alone. Writes `comparison_per_country.csv`, `comparison_overall.csv` and `figures/comparison.png` here.
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
# label -> (experiment folder, setting, seed filter or None = the summary's mean over seeds)
RUNS = {
    "TabICL default": ("exp_1_eu_grid_dense_omfix", "tabicl:tabicl_default_single", None),
    "SCM control (EU grid shapes)": ("exp_1_eu_grid", "tabicl_wofost:scm", None),
    "WOFOST, exp_1 points": ("exp_1", W, None),
    "WOFOST, EU grid": ("exp_1_eu_grid", W, None),
    "WOFOST, densified, old soil": ("exp_1_eu_grid_dense", W, None),
    "WOFOST, densified, OM fix": ("exp_1_eu_grid_dense_omfix", W, None),
    "A: log-uniform points (3-seed mean)": ("exp_2a_loguniform_points", W, None),
    "A: log-uniform points (seed 0)": ("exp_2a_loguniform_points", W, 0.0),
    "C: irrigation map (seed 0)": ("exp_3_irrigation_map", W, None),
    "D: observational noise (seed 0)": ("exp_4_obs_noise", W, None),
}
COLORS = ["#7a7a7a", "#f2c9a0", "#c9b8e8", "#8f7cc4", "#7fcdb5", "#1b9e77", "#f4a261", "#d95f02", "#2a78d6", "#b3122e"]


def load(exp: str, setting: str, seed) -> pd.DataFrame:
    if seed is None:
        s = pd.read_csv(f"{BENCH}/{exp}/summary.csv")
        return s[s.setting == setting]
    s = pd.read_csv(f"{BENCH}/{exp}/per_run.csv")  # one row per seed
    return s[(s.setting == setting) & (s.seed == seed)]


def main():
    frames = []
    for label, (exp, setting, seed) in RUNS.items():
        if not os.path.exists(f"{BENCH}/{exp}/summary.csv"):
            print(f"missing: {label}")
            continue
        s = load(exp, setting, seed)
        frames.append(s[["country", "nrmse", "anomaly_r", "r2", "n_region_years"]].assign(run=label))
    df = pd.concat(frames, ignore_index=True)
    runs = [r for r in RUNS if r in set(df.run)]
    per_country = df.pivot(index="country", columns="run", values=["nrmse", "anomaly_r"]).reindex(columns=runs, level=1)
    per_country.round(4).to_csv(f"{OUT}/comparison_per_country.csv")
    rows = []
    for run in runs:
        g = df[df.run == run]
        rows.append({"run": run, "n_countries": len(g), "nrmse_mean": g.nrmse.mean(), "nrmse_weighted": np.average(g.nrmse, weights=g.n_region_years),
                     "anomaly_r_mean": g.anomaly_r.mean(), "anomaly_r_weighted": np.average(g.anomaly_r, weights=g.n_region_years)})
    overall = pd.DataFrame(rows).set_index("run")
    overall.round(4).to_csv(f"{OUT}/comparison_overall.csv")
    print(overall.round(4).to_string())
    ref = "A: log-uniform points (seed 0)"
    for new in ("C: irrigation map (seed 0)", "D: observational noise (seed 0)"):
        if new in runs and ref in runs:
            d, a = per_country["nrmse"][new] - per_country["nrmse"][ref], per_country["anomaly_r"][new] - per_country["anomaly_r"][ref]
            print(f"{new} vs {ref}: NRMSE lower in {(d < 0).sum()} of {len(d)} countries (mean diff {d.mean():+.4f}); "
                  f"anomaly r higher in {(a > 0).sum()} (mean diff {a.mean():+.4f})")

    countries = per_country.index.tolist()
    fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=True)
    n = len(runs)
    x, w = np.arange(len(countries)), 0.84 / n
    for ax, metric, title in ((axes[0], "nrmse", "NRMSE (lower is better)"), (axes[1], "anomaly_r", "anomaly r (higher is better)")):
        for i, run in enumerate(runs):
            ax.bar(x + (i - (n - 1) / 2) * w, per_country[metric][run], w, color=COLORS[list(RUNS).index(run)], label=run)
        ax.set_ylabel(title)
    axes[1].set_xticks(x, countries)
    axes[0].legend(fontsize=7, ncol=5, frameon=False)
    fig.suptitle("CY-Bench walk-forward, 18 European maize countries: irrigation map (C) and observational noise (D) vs earlier runs "
                 "(C and D: one pretraining seed)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/figures/comparison.png", dpi=140)


if __name__ == "__main__":
    main()
