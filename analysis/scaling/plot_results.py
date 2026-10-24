"""exp_1 on the EU grid WOFOST table (exp_1_eu_grid), and against the original exp_1 (FR/BE/DK table).

    uv run python -m analysis.scaling.plot_results     # after the exp_1_eu_grid summary job

Reads the benchmark's per_run.csv of both experiments (`summarize_exp.py`) and writes to
`analysis/scaling/figures/`:

- `results_overview.png`: exp_1_eu_grid, default TabICL vs SCM control vs WOFOST, over the 18 countries;
- `results_per_country.png`: exp_1_eu_grid, NRMSE minus default TabICL's, per country;
- `results_vs_previous.png`: both experiments side by side (overview + per-country change vs TabICL).

It also copies the per_run.csv to `analysis/scaling/results_exp1_eu_grid_per_run.csv`.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.plot_results import INK, INK2, REF, SCM, WOFOST

BENCH = Path("/projects/prjs1788/tnp-crop/cybench_output")
HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
TABICL, SCM_SET, WOFOST_SET = "tabicl:tabicl_default_single", "tabicl_wofost:scm", "tabicl_wofost:wofost"
# (experiment, setting) -> (label, color, marker, filled)
NEW = {(None, TABICL): ("TabICL default", REF, "D", True),
       ("eu_grid", SCM_SET): ("SCM control", SCM, "s", True),
       ("eu_grid", WOFOST_SET): ("WOFOST", WOFOST, "o", True)}
BOTH = {(None, TABICL): ("TabICL default", REF, "D", True),
        ("fr_be_dk", SCM_SET): ("SCM control, shapes of FR/BE/DK table", SCM, "s", False),
        ("eu_grid", SCM_SET): ("SCM control, shapes of EU grid table", SCM, "s", True),
        ("fr_be_dk", WOFOST_SET): ("WOFOST, FR/BE/DK table (exp_1)", WOFOST, "o", False),
        ("eu_grid", WOFOST_SET): ("WOFOST, EU grid table", WOFOST, "o", True)}


def runs() -> pd.DataFrame:
    frames = []
    for exp, tag in (("exp_1_eu_grid", "eu_grid"), ("exp_1", "fr_be_dk")):
        r = pd.read_csv(BENCH / exp / "per_run.csv").assign(exp=tag)
        r["seed"] = r["seed"].fillna(0).astype(int)
        frames.append(r)
    out = pd.concat(frames, ignore_index=True)
    out.loc[out["setting"] == TABICL, "exp"] = None  # deterministic: the same in both experiments
    return out.drop_duplicates(["exp", "setting", "seed", "country"])


def _select(r, key):
    exp, setting = key
    return r[(r["setting"] == setting) & (r["exp"].isna() if exp is None else r["exp"] == exp)]


def _marker(ax, x, y, color, marker, filled, size, **kw):
    ax.scatter(x, y, s=size, marker=marker, color=color if filled else "white", edgecolors=color,
               linewidths=1.5, **kw)


def overview(r, settings, name, title):
    def per_seed(g):
        w = g["n_region_years"]
        return pd.Series({"nrmse": g["nrmse"].mean(), "nrmse_w": (w * g["nrmse"]).sum() / w.sum(),
                          "temporal_r": g["temporal_r"].mean(), "anomaly_r": g["anomaly_r"].mean()})
    panels = [("nrmse", "NRMSE\nmean over countries", "lower is better"),
              ("nrmse_w", "NRMSE\nweighted by region-years", "lower is better"),
              ("temporal_r", "Temporal r\nmean over countries", "higher is better"),
              ("anomaly_r", "Anomaly r\nmean over countries", "higher is better")]
    keys = list(settings)
    fig, axes = plt.subplots(1, 4, figsize=(12, 0.45 * len(keys) + 1.6), sharey=True)
    for ax, (col, ptitle, hint) in zip(axes, panels):
        for y, key in enumerate(keys):
            _, color, marker, filled = settings[key]
            agg = _select(r, key).groupby("seed").apply(per_seed, include_groups=False)[col].to_numpy()
            ax.scatter(agg, np.full(len(agg), y), s=20, color=color, alpha=0.4, marker=marker, linewidths=0)
            _marker(ax, [agg.mean()], [y], color, marker, filled, 70, zorder=3)
            ax.annotate(f"{agg.mean():.3f}", (agg.mean(), y), xytext=(0, 7), textcoords="offset points",
                        ha="center", fontsize=8, color=INK2)
        ax.set_title(ptitle, loc="left")
        ax.set_xlabel(hint)
        ax.grid(axis="y", visible=False)
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo - 0.15 * (hi - lo), hi + 0.15 * (hi - lo))
    axes[0].set_yticks(range(len(keys)), [settings[k][0] for k in keys])
    axes[0].set_ylim(len(keys) - 0.4, -0.6)
    fig.suptitle(title, x=0.0, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)


def per_country(r, settings, name, title):
    ref = _select(r, (None, TABICL)).set_index("country")["nrmse"]
    n = r.groupby("country")["n_region_years"].first()
    keys = [k for k in settings if k[1] != TABICL]
    stats = {k: _select(r, k).groupby("country")["nrmse"].agg(["mean", "std"]) for k in keys}
    sort_key = ("eu_grid", WOFOST_SET)
    countries = (stats[sort_key]["mean"] - ref).sort_values().index
    y = np.arange(len(countries))
    offsets = np.linspace(-0.3, 0.3, len(keys))
    fig, ax = plt.subplots(figsize=(6.6, 0.38 * len(countries) + 1.6))
    ax.axvline(0, color=INK2, linewidth=1, zorder=1)
    for key, off in zip(keys, offsets):
        label, color, marker, filled = settings[key]
        s = stats[key].loc[countries]
        d = s["mean"] - ref.loc[countries]
        ax.errorbar(d, y + off, xerr=2 * s["std"], fmt="none", ecolor=color, elinewidth=1.3, zorder=2)
        _marker(ax, d, y + off, color, marker, filled, 34, zorder=3, label=label)
    ax.set_yticks(y, [f"{c}  (n={n[c]})" for c in countries])
    ax.set_ylim(len(countries) - 0.5, -0.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("NRMSE − NRMSE of default TabICL\n← better than TabICL          worse than TabICL →")
    ax.set_title(title, loc="left", pad=22)
    ax.text(0, 1.01, "mean over 3 pretraining seeds, bars = ±2 sd over seeds; n = test region-years",
            transform=ax.transAxes, fontsize=8, color=INK2, va="bottom")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.1 - 3.0 / (0.38 * len(countries) + 1.6) / 10), ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    r = runs()
    pd.read_csv(BENCH / "exp_1_eu_grid" / "per_run.csv").to_csv(HERE / "results_exp1_eu_grid_per_run.csv", index=False)
    single = "18 European maize countries, single pass\n(large marker = mean over 3 pretraining seeds, small = each seed)"
    overview(r, NEW, "results_overview", f"exp_1 on the EU grid WOFOST table: {single}")
    per_country(r, NEW, "results_per_country", "exp_1 on the EU grid table: change in NRMSE vs default TabICL")
    overview(r, BOTH, "results_vs_previous_overview",
             f"EU grid table (filled) vs FR/BE/DK table of exp_1 (hollow): {single}")
    per_country(r, BOTH, "results_vs_previous_per_country",
                "EU grid (filled) vs FR/BE/DK table (hollow): NRMSE vs default TabICL")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
