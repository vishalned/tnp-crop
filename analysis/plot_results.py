"""Figures for docs/results.md, from the result CSVs in docs/ (no benchmark run needed).

    uv run python -m analysis.plot_results            # writes analysis/figures/*.png

1. fig1_exp1_overview      exp_1, 18 European countries: mean / weighted NRMSE, temporal r, anomaly r
2. fig2_exp1_per_country   exp_1: NRMSE difference to default TabICL per country (WOFOST, SCM control)
3. fig3_lr_test            LR test (DE/FR/BE): NRMSE vs learning rate, screening and walk-forward
4. fig4_early_runs         first runs (DE/FR/BE, step 500): run-to-run spread of WOFOST vs references
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
OUT = Path(__file__).resolve().parent / "figures"

# palette: categorical slots 1-2 of the reference palette; the reference model in neutral gray
WOFOST, SCM, REF = "#2a78d6", "#eb6834", "#8a8984"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
SETTINGS = {  # summarize_exp.py setting -> (label, color, marker)
    "tabicl:tabicl_default_single": ("TabICL default", REF, "D"),
    "tabicl_wofost:scm": ("SCM control", SCM, "s"),
    "tabicl_wofost:wofost": ("WOFOST", WOFOST, "o"),
}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight", "font.size": 9,
    "axes.edgecolor": INK2, "axes.labelcolor": INK2, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False, "figure.facecolor": "white",
})


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print(f"wrote {OUT / name}.png")


# ---------------------------------------------------------------- exp_1

def exp1_runs() -> pd.DataFrame:
    runs = pd.read_csv(DOCS / "results_exp1_per_run.csv")
    runs["seed"] = runs["seed"].fillna(0).astype(int)  # default TabICL: one deterministic run
    return runs


def fig1_exp1_overview(runs):
    """One value per (setting, pretraining seed) aggregated over the 18 countries; dots = seeds."""
    def per_seed(g):
        w = g["n_region_years"]
        return pd.Series({
            "nrmse": g["nrmse"].mean(),
            "nrmse_w": (w * g["nrmse"]).sum() / w.sum(),
            "temporal_r": g["temporal_r"].mean(),
            "anomaly_r": g["anomaly_r"].mean(),
        })
    agg = runs.groupby(["setting", "seed"]).apply(per_seed, include_groups=False).reset_index()
    panels = [("nrmse", "NRMSE\nmean over countries", "lower is better"),
              ("nrmse_w", "NRMSE\nweighted by region-years", "lower is better"),
              ("temporal_r", "Temporal r\nmean over countries", "higher is better"),
              ("anomaly_r", "Anomaly r\nmean over countries", "higher is better")]
    order = list(SETTINGS)
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.8), sharey=True)
    for ax, (col, title, hint) in zip(axes, panels):
        for y, setting in enumerate(order):
            label, color, marker = SETTINGS[setting]
            vals = agg.loc[agg["setting"] == setting, col].to_numpy()
            ax.scatter(vals, np.full(len(vals), y), s=22, color=color, alpha=0.45, marker=marker,
                       linewidths=0, zorder=2)
            ax.scatter([vals.mean()], [y], s=70, color=color, marker=marker,
                       edgecolors="white", linewidths=1.5, zorder=3)
            ax.annotate(f"{vals.mean():.3f}", (vals.mean(), y), xytext=(0, 7), textcoords="offset points",
                        ha="center", fontsize=8, color=INK2)
        ax.set_title(title, loc="left")
        ax.set_xlabel(hint)
        ax.grid(axis="y", visible=False)
        lo, hi = ax.get_xlim()
        pad = 0.15 * (hi - lo)
        ax.set_xlim(lo - pad, hi + pad)
    axes[0].set_yticks(range(len(order)), [SETTINGS[s][0] for s in order])
    axes[0].set_ylim(-0.6, len(order) - 0.4)
    fig.suptitle("exp_1: 18 European maize countries, single pass "
                 "(large marker = mean over 3 pretraining seeds, small = each seed)",
                 x=0.0, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    save(fig, "fig1_exp1_overview")


def fig2_exp1_per_country(runs):
    """NRMSE - NRMSE(default TabICL) per country; error bars = 2 sd over pretraining seeds."""
    ref = runs[runs["setting"] == "tabicl:tabicl_default_single"].set_index("country")["nrmse"]
    n = runs.groupby("country")["n_region_years"].first()
    stats = {}
    for setting in ("tabicl_wofost:wofost", "tabicl_wofost:scm"):
        g = runs[runs["setting"] == setting].groupby("country")["nrmse"]
        stats[setting] = pd.DataFrame({"d": g.mean() - ref, "sd": g.std()})
    countries = stats["tabicl_wofost:wofost"]["d"].sort_values().index  # largest WOFOST gain on top
    y = np.arange(len(countries))
    fig, ax = plt.subplots(figsize=(6.2, 6.4))
    ax.axvline(0, color=INK2, linewidth=1, zorder=1)
    for setting, offset in (("tabicl_wofost:wofost", -0.17), ("tabicl_wofost:scm", 0.17)):
        label, color, marker = SETTINGS[setting]
        s = stats[setting].loc[countries]
        ax.errorbar(s["d"], y + offset, xerr=2 * s["sd"], fmt=marker, color=color, ms=6,
                    elinewidth=1.5, capsize=0, mec="white", mew=0.8, label=label, zorder=3)
    ax.set_yticks(y, [f"{c}  (n={n[c]})" for c in countries])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("NRMSE − NRMSE of default TabICL\n← better than TabICL          worse than TabICL →")
    ax.set_title("exp_1: change in NRMSE vs default TabICL, per country", loc="left", pad=20)
    ax.text(0, 1.01, "mean over 3 pretraining seeds, bars = ±2 sd over seeds; n = test region-years",
            transform=ax.transAxes, fontsize=8, color=INK2, va="bottom")
    ax.legend(loc="upper right")
    fig.tight_layout()
    save(fig, "fig2_exp1_per_country")


# ---------------------------------------------------------------- LR test

def fig3_lr_test():
    lr = pd.read_csv(DOCS / "results_lr_test.csv")
    parsed = lr["run"].str.extract(r"_(?P<data>wofost|scm)_lr(?P<lr>[0-9.e-]+)_(?P<mode>ens8|single)$")
    lr = lr.join(parsed)
    lr["lr"] = pd.to_numeric(lr["lr"])
    refs = {"tabicl_eos": ("TabICL default, 1 pass", "-"), "tabicl_eos_default": ("TabICL default, 8 est.", ":")}
    countries = ["DE", "FR", "BE"]
    rows = [("scr_nrmse", "screening (val. years)"), ("wf_nrmse", "walk-forward (test years)")]
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.4), sharex=True)
    for r, (col, rlabel) in enumerate(rows):
        for c, country in enumerate(countries):
            ax = axes[r, c]
            d = lr[lr["country"] == country]
            for run, (_, ls) in refs.items():
                v = d.loc[d["run"] == run, col]
                if len(v):
                    ax.axhline(v.iloc[0], color=REF, linestyle=ls, linewidth=1.3, zorder=1)
            for data, color in (("wofost", WOFOST), ("scm", SCM)):
                for mode, ls, fill in (("single", "-", color), ("ens8", "--", "white")):
                    s = d[(d["data"] == data) & (d["mode"] == mode)].sort_values("lr")
                    ax.plot(s["lr"], s[col], ls, color=color, linewidth=2, marker="o", ms=6,
                            mfc=fill, mec=color, mew=1.5, zorder=3)
            ax.set_xscale("log")
            ax.set_xticks([3e-6, 1e-5, 3e-5], ["3e-6", "1e-5", "3e-5"])
            ax.minorticks_off()
            if r == 0:
                ax.set_title(country, loc="left")
            if c == 0:
                ax.set_ylabel(f"NRMSE\n{rlabel}")
            if r == 1:
                ax.set_xlabel("learning rate")
    handles = [Line2D([], [], color=WOFOST, lw=2, label="WOFOST"),
               Line2D([], [], color=SCM, lw=2, label="SCM control"),
               Line2D([], [], color=INK2, lw=2, marker="o", label="1 pass"),
               Line2D([], [], color=INK2, lw=2, ls="--", marker="o", mfc="white", label="8 estimators"),
               Line2D([], [], color=REF, lw=1.3, label=refs["tabicl_eos"][0]),
               Line2D([], [], color=REF, lw=1.3, ls=":", label=refs["tabicl_eos_default"][0])]
    fig.legend(handles=handles, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("LR test: 3000 steps, cosine to 0, last checkpoint (lower is better)",
                 x=0.0, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "fig3_lr_test")


# ---------------------------------------------------------------- first runs

def fig4_early_runs():
    pooled = pd.read_csv(DOCS / "results_cybench_pooled.csv")
    rows = [  # (label, model prefix in the CSV, color)
        ("TabICL default", "TabICL default", REF),
        ("NanoTabICL + official weights", "NanoTabICL + official weights, fixed grouping", REF),
        ("WOFOST run 1 (old grouping)", "NanoTabICL-WOFOST run 1", WOFOST),
        ("WOFOST run 2 (rerun)", "NanoTabICL-WOFOST run 2", WOFOST),
        ("SCM control", "NanoTabICL-SCM control", SCM),
    ]
    countries = ["DE", "FR", "BE"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.8), sharey=True)
    for ax, country in zip(axes, countries):
        d = pooled[pooled["country"] == country]
        for y, (label, prefix, color) in enumerate(rows):
            m = d[d["model"].str.startswith(prefix)]
            ens = m[m["model"].str.contains("8 estimators")]["nrmse"]
            single = m[m["model"].str.contains("1 estimator|1 pass")]["nrmse"]
            if len(ens) and len(single):
                ax.plot([single.iloc[0], ens.iloc[0]], [y, y], color=color, alpha=0.35, lw=2, zorder=1)
            ax.scatter(single, [y] * len(single), s=40, color=color, edgecolor="white", lw=0.8, zorder=3)
            ax.scatter(ens, [y] * len(ens), s=40, facecolor="white", edgecolor=color, lw=1.5, zorder=2)
        ax.set_title(f"{country}", loc="left")
        ax.set_xlabel("walk-forward NRMSE (lower is better)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(rows)), [r[0] for r in rows])
    axes[0].invert_yaxis()
    handles = [Line2D([], [], ls="", marker="o", color=INK2, label="1 pass"),
               Line2D([], [], ls="", marker="o", mfc="white", mec=INK2, mew=1.5, label="8 estimators")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.6, -0.06))
    fig.suptitle("First runs (step 500, old recipe): two WOFOST runs with the same setup differ a lot",
                 x=0.0, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "fig4_early_runs")


def main():
    runs = exp1_runs()
    fig1_exp1_overview(runs)
    fig2_exp1_per_country(runs)
    fig3_lr_test()
    fig4_early_runs()


if __name__ == "__main__":
    main()
