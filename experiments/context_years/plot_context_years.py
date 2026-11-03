"""Does the pretraining help when the context is short?  (exp_5_context_years)

    uv run python -m experiments.context_years.plot_context_years

CY-Bench maize walk-forward (single pass, pretraining seed 0 where it applies). For DE, FR, HU, ES the context of every forecast
year is cut to the last 2 / 3 / 5 years before it (all regions kept); "all" = every earlier year (results of the earlier runs).
Senegal (SN) is a side check with all years. Writes context_years.csv and figures/context_years.png, figures/senegal.png.
"""
import os

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUNS = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results", "runs.csv"))  # all results, collected by analysis.collect_results
OUT = os.path.dirname(os.path.abspath(__file__))
COUNTRIES = ["DE", "FR", "HU", "ES"]
LEVELS = ["2", "3", "5", "all"]
# label -> (tag base in exp_5, setting+folder of the existing "all years" result, colour)
MODELS = {
    "TabICL default": ("tabicl_default_single", ("exp_1_eu_grid_dense_omfix", "tabicl:tabicl_default_single", None), "#7a7a7a"),
    "SCM control": ("scm", ("exp_1_eu_grid", "tabicl_wofost:scm", 0.0), "#f2c9a0"),
    "WOFOST (densified OM fix, log-uniform)": ("wofost", ("exp_2a_loguniform_points", "tabicl_wofost:wofost", 0.0), "#1b9e77"),
    "+ irrigation map": ("wofost_irrmap", ("exp_3_irrigation_map", "tabicl_wofost:wofost", None), "#2a78d6"),
}


def get(exp, setting, seed):
    d = RUNS[(RUNS.experiment == exp) & (RUNS.setting == setting)]
    return d if seed is None else d[d.seed == seed]


def main():
    new = RUNS[RUNS.experiment == "exp_5_context_years"]
    rows = []
    for label, (tag, (exp, setting, seed), _) in MODELS.items():
        for k in LEVELS:
            if k == "all":
                d = get(exp, setting, seed)
                d = d[d.country.isin(COUNTRIES)].assign(ctx="all")
                sn = new[(new.country == "SN") & (new.setting.str.endswith(f"{tag}_ctxall"))].assign(ctx="all")
                d = pd.concat([d, sn])
            else:
                d = new[(new.country.isin(COUNTRIES)) & (new.setting.str.endswith(f"{tag}_ctx{k}"))].assign(ctx=k)
            rows.append(d.assign(model=label)[["country", "ctx", "model", "nrmse", "anomaly_r", "r2", "n_region_years"]])
    df = pd.concat(rows, ignore_index=True)
    df.round(4).to_csv(f"{OUT}/context_years.csv", index=False)

    fig, axes = plt.subplots(2, 5, figsize=(17, 6.5), sharex="col")
    for j, c in enumerate(COUNTRIES + ["mean"]):
        for i, (m, title) in enumerate([("nrmse", "NRMSE (lower is better)"), ("anomaly_r", "anomaly r (higher is better)")]):
            ax = axes[i, j]
            for label, (_, _, col) in MODELS.items():
                d = df[df.model == label]
                d = d[d.country.isin(COUNTRIES)]
                s = (d.groupby("ctx")[m].mean() if c == "mean" else d[d.country == c].set_index("ctx")[m]).reindex(LEVELS)
                ax.plot(LEVELS, s.values, "-o", color=col, label=label, lw=2)
            if i == 0:
                ax.set_title(c if c != "mean" else "mean of the 4")
            if j == 0:
                ax.set_ylabel(title)
            ax.grid(alpha=0.3)
            ax.set_xlabel("context years")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Short-context test: CY-Bench maize, walk-forward, context = last k years", y=1.0)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figures/context_years.png", dpi=150)

    sn = df[(df.country == "SN") & (df.ctx == "all")]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, m in zip(axes, ["nrmse", "anomaly_r"]):
        ax.bar(range(len(sn)), sn[m], color=[MODELS[x][2] for x in sn.model])
        ax.set_xticks(range(len(sn)), [x.split(" (")[0] for x in sn.model], rotation=20, ha="right", fontsize=8)
        ax.set_title(f"Senegal, all years: {m}")
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figures/senegal.png", dpi=150)
    print(df.pivot_table(index=["model", "ctx"], columns="country", values="nrmse").round(3).to_string())


if __name__ == "__main__":
    main()
