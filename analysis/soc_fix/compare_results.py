"""CY-Bench walk-forward results (18 European maize countries, single pass) of the OM-fix run against the earlier runs.

    uv run python -m analysis.soc_fix.compare_results

Reads each experiment's `summary.csv` (mean over the 3 pretraining seeds per country) from
`/projects/prjs1788/tnp-crop/cybench_output/` and writes `analysis/soc_fix/comparison_per_country.csv`,
`comparison_overall.csv` (mean over countries, and weighted by region-years) and `figures/comparison.png`.
"""

import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BENCH = "/projects/prjs1788/tnp-crop/cybench_output"
OUT = os.path.dirname(os.path.abspath(__file__))
# label -> (experiment folder, setting in its summary.csv)
RUNS = {
    "TabICL default": ("exp_1_eu_grid_dense_omfix", "tabicl:tabicl_default_single"),
    "WOFOST, exp_1 (original points)": ("exp_1", "tabicl_wofost:wofost"),
    "SCM control, exp_1": ("exp_1", "tabicl_wofost:scm"),
    "WOFOST, EU grid": ("exp_1_eu_grid", "tabicl_wofost:wofost"),
    "SCM control, EU grid": ("exp_1_eu_grid", "tabicl_wofost:scm"),
    "WOFOST, densified EU grid (old soil)": ("exp_1_eu_grid_dense", "tabicl_wofost:wofost"),
    "WOFOST, densified EU grid, OM fix": ("exp_1_eu_grid_dense_omfix", "tabicl_wofost:wofost"),
}
COLORS = ["#7a7a7a", "#c9b8e8", "#f2c9a0", "#8f7cc4", "#d95f02", "#7fcdb5", "#1b9e77"]


def main():
    frames = []
    for label, (exp, setting) in RUNS.items():
        s = pd.read_csv(f"{BENCH}/{exp}/summary.csv")
        s = s[s.setting == setting].assign(run=label)
        frames.append(s[["run", "country", "nrmse", "nrmse_sd", "anomaly_r", "anomaly_r_sd", "r2", "n_region_years"]])
    df = pd.concat(frames, ignore_index=True)
    per_country = df.pivot(index="country", columns="run", values=["nrmse", "anomaly_r"]).reindex(columns=list(RUNS), level=1)
    per_country.round(4).to_csv(f"{OUT}/comparison_per_country.csv")

    rows = []
    for run, g in df.groupby("run", sort=False):
        w = g.n_region_years
        rows.append({"run": run, "n_countries": len(g), "nrmse_mean": g.nrmse.mean(), "nrmse_weighted": np.average(g.nrmse, weights=w),
                     "anomaly_r_mean": g.anomaly_r.mean(), "anomaly_r_weighted": np.average(g.anomaly_r, weights=w)})
    overall = pd.DataFrame(rows).set_index("run")
    # paired difference to the old-soil densified run: per country, then mean (and how many countries improve)
    old, new = per_country["nrmse"]["WOFOST, densified EU grid (old soil)"], per_country["nrmse"]["WOFOST, densified EU grid, OM fix"]
    overall.loc["OM fix minus old soil (densified)", ["nrmse_mean"]] = (new - old).mean()
    overall.round(4).to_csv(f"{OUT}/comparison_overall.csv")
    print(overall.round(4).to_string())
    print(f"OM fix better (lower NRMSE) than old soil in {(new < old).sum()} of {len(new)} countries; "
          f"anomaly r higher in {(per_country['anomaly_r']['WOFOST, densified EU grid, OM fix'] > per_country['anomaly_r']['WOFOST, densified EU grid (old soil)']).sum()}")

    countries = per_country.index.tolist()
    fig, axes = plt.subplots(2, 1, figsize=(15, 7.5), sharex=True)
    x, w = np.arange(len(countries)), 0.11
    for ax, metric, title in ((axes[0], "nrmse", "NRMSE (lower is better)"), (axes[1], "anomaly_r", "anomaly r (higher is better)")):
        for i, (run, c) in enumerate(zip(RUNS, COLORS)):
            ax.bar(x + (i - 3) * w, per_country[metric][run], w, color=c, label=run)
        ax.set_ylabel(title)
    axes[1].set_xticks(x, countries)
    axes[0].legend(fontsize=7, ncol=4, frameon=False)
    fig.suptitle("CY-Bench walk-forward, 18 European maize countries: OM-fix run vs earlier runs (mean over pretraining seeds)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/figures/comparison.png", dpi=140)


if __name__ == "__main__":
    main()
