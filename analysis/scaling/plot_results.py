"""exp_1 on the EU grid WOFOST table (exp_1_eu_grid), and against the original exp_1 (FR/BE/DK table).

    uv run python -m analysis.scaling.plot_results     # after the exp_1_eu_grid summary job

Reads the benchmark's per_run.csv of both experiments (`summarize_exp.py`) and writes to
`analysis/scaling/figures/`:

- `results_overview.png`: exp_1_eu_grid, default TabICL vs SCM control vs WOFOST, over the 18 countries;
- `results_per_country.png`: exp_1_eu_grid, NRMSE minus default TabICL's, per country;
- `results_vs_previous_*.png`: both experiments side by side (overview + per-country change vs TabICL);
- `decisive_test.png`: old table, old table subsampled to 466 locations, new table FR/BE/DK only, new table;
- `results_all_models_overview.png` / `_heatmap.png`: every model on the same test rows: default TabICL,
  the EU grid run's SCM control, WOFOST pretraining on the FR/BE/DK table (exp_1), the EU grid table and
  (once run) the densified EU grid table (`docs/country_share_report.md`), and the uncalibrated WOFOST
  yield as a baseline (`analysis/wofost_baseline_cybench`, scored by the benchmark's summarize_exp.py).
  Overview = means over the 18 countries; heatmap = per country x model, NRMSE and anomaly r.

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


def runs(experiments=(("exp_1_eu_grid", "eu_grid"), ("exp_1", "fr_be_dk"))) -> pd.DataFrame:
    frames = []
    for exp, tag in experiments:
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


def per_country(r, settings, name, title, sort_key=("eu_grid", WOFOST_SET)):
    ref = _select(r, (None, TABICL)).set_index("country")["nrmse"]
    n = r.groupby("country")["n_region_years"].first()
    keys = [k for k in settings if k[1] != TABICL]
    stats = {k: _select(r, k).groupby("country")["nrmse"].agg(["mean", "std"]) for k in keys}
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



# ---------------------------------------------------------------- decisive test (old vs new pipeline)

STEPS = [("exp_1", "old table, FR/BE/DK, 1,563 locations (exp_1)"),
         ("exp_1_old_matched_466", "old table, subsampled to 421/22/23 locations"),
         ("exp_1_eu_grid_frbedk", "new EU grid table, FR/BE/DK only (421/22/23)"),
         ("exp_1_eu_grid", "new EU grid table, 18 countries (1,966)")]


def decisive_test():
    """WOFOST NRMSE vs default TabICL for the four training tables: overall and per country."""
    frames = {}
    for exp, _ in STEPS:
        r = pd.read_csv(BENCH / exp / "per_run.csv")
        r["seed"] = r["seed"].fillna(0).astype(int)
        frames[exp] = r
    ref = frames["exp_1"].query("setting == @TABICL").set_index("country")["nrmse"]
    n = frames["exp_1"].groupby("country")["n_region_years"].first()
    shades = ["#9ec5f4", "#5f9de6", "#2a78d6", "#14498a"]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 6.4), gridspec_kw={"width_ratios": [1, 1.3]})
    for k, ((exp, label), color) in enumerate(zip(STEPS, shades)):
        r = frames[exp]
        for setting, marker, c in ((WOFOST_SET, "o", color), (SCM_SET, "s", SCM)):
            per_seed = r[r["setting"] == setting].groupby("seed")["nrmse"].mean()
            ax0.scatter(per_seed, np.full(len(per_seed), k + (0.15 if setting == SCM_SET else -0.15)), s=16,
                        color=c, alpha=0.5, marker=marker, linewidths=0)
            ax0.scatter([per_seed.mean()], [k + (0.15 if setting == SCM_SET else -0.15)], s=60, color=c,
                        marker=marker, edgecolors="white", linewidths=1, zorder=3,
                        label="SCM control" if (setting == SCM_SET and k == 0) else None)
            if setting == WOFOST_SET:
                ax0.annotate(f"{per_seed.mean():.4f}", (per_seed.mean(), k - 0.15), xytext=(0, -12),
                             textcoords="offset points", ha="center", fontsize=8, color=INK2)
        d = r[r["setting"] == WOFOST_SET].groupby("country")["nrmse"].mean() - ref
        countries = n.sort_values(ascending=False).index
        ax1.scatter(d.loc[countries], np.arange(len(countries)) + (k - 1.5) * 0.18, s=26, color=color,
                    edgecolors="white", linewidths=0.5, zorder=3, label=label)
    ax0.axvline(ref.mean(), color=REF, lw=1.5, ls="--", label=f"default TabICL ({ref.mean():.4f})")
    ax0.set_yticks(range(len(STEPS)), [lbl for _, lbl in STEPS])
    ax0.set_ylim(len(STEPS) - 0.5, -0.5)
    ax0.set_xlabel("NRMSE, mean over 18 countries (lower is better)")
    ax0.set_title("WOFOST (circles) and SCM control (squares), 3 seeds", loc="left")
    ax0.legend(loc="lower right", fontsize=8)
    ax0.grid(axis="y", visible=False)
    ax1.axvline(0, color=INK2, lw=1)
    ax1.set_yticks(np.arange(len(countries)), [f"{c} (n={n[c]})" for c in countries])
    ax1.set_ylim(len(countries) - 0.5, -0.5)
    ax1.grid(axis="y", visible=False)
    ax1.set_xlabel("WOFOST NRMSE − default TabICL NRMSE (mean of 3 seeds)\n← better than TabICL")
    ax1.set_title("per country, the four training tables (light → dark)", loc="left")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.45, -0.12), ncol=2, fontsize=8)
    fig.suptitle("Where the WOFOST gain went: table size, location choice, added countries",
                 x=0.0, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "decisive_test.png")
    plt.close(fig)
    print(f"wrote {OUT / 'decisive_test.png'}")


# ---------------------------------------------------------------- all models: FR/BE/DK -> EU grid -> densified, + baseline

BASELINE = ("wofost_baseline", "baseline")  # analysis/wofost_baseline_cybench --test, scored by summarize_exp.py
BASELINE_SET = "wofost:raw_eu_grid"
ALL_RUNS = (("exp_1", "fr_be_dk"), ("exp_1_eu_grid", "eu_grid"), ("exp_1_eu_grid_dense", "eu_grid_dense"), BASELINE)
ALL = {(None, TABICL): ("TabICL default", REF, "D", True),
       ("eu_grid", SCM_SET): ("SCM control (EU grid shapes)", SCM, "s", True),
       ("fr_be_dk", WOFOST_SET): ("WOFOST pretraining, FR/BE/DK table (exp_1)", "#9ec5f4", "o", True),
       ("eu_grid", WOFOST_SET): ("WOFOST pretraining, EU grid table", "#2a78d6", "o", True),
       ("eu_grid_dense", WOFOST_SET): ("WOFOST pretraining, densified EU grid table", "#14498a", "o", True),
       ("baseline", BASELINE_SET): ("WOFOST yield, uncalibrated (WLP-MLWB, EU grid points)", "#7f3c8d", "v", True)}
SHORT = {(None, TABICL): "TabICL\ndefault", ("eu_grid", SCM_SET): "SCM\ncontrol", ("fr_be_dk", WOFOST_SET): "WOFOST\nFR/BE/DK",
         ("eu_grid", WOFOST_SET): "WOFOST\nEU grid", ("eu_grid_dense", WOFOST_SET): "WOFOST\ndensified",
         ("baseline", BASELINE_SET): "raw WOFOST\nyield"}


def heatmap(r, settings, name, title):
    """Countries x models: the value in each cell, coloured by its difference to default TabICL."""
    from matplotlib.colors import TwoSlopeNorm
    keys = list(settings)
    n = r.groupby("country")["n_region_years"].max()
    countries = list(n.sort_values(ascending=False).index)
    panels = [("nrmse", "NRMSE (lower is better)", -1), ("anomaly_r", "anomaly r (higher is better)", 1)]
    fig, axes = plt.subplots(1, 2, figsize=(2 * (1.05 * len(keys) + 1.2), 0.3 * len(countries) + 2.4))
    for ax, (col, ptitle, sign) in zip(axes, panels):
        m = pd.DataFrame({k: _select(r, k).groupby("country")[col].mean() for k in keys}).reindex(countries)
        m.loc["mean"] = m.mean()
        better = sign * (m.sub(m[(None, TABICL)], axis=0))  # > 0: better than default TabICL
        lim = 0.05 if col == "nrmse" else 0.15
        ax.imshow(better.clip(-lim, lim).to_numpy(), cmap="RdBu", norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
        for (i, j), v in np.ndenumerate(m.to_numpy()):
            ax.text(j, i, "" if np.isnan(v) else f"{v:.3f}" if col == "nrmse" else f"{v:.2f}", ha="center",
                    va="center", fontsize=6.8, color="white" if abs(better.iat[i, j]) > 0.75 * lim else INK,
                    fontweight="bold" if i == len(m) - 1 else None)
        ax.axhline(len(countries) - 0.5, color=INK, lw=1.2)
        ax.set_xticks(range(len(keys)), [SHORT[k] for k in keys], fontsize=7.5)
        ax.xaxis.tick_top()
        ax.set_yticks(range(len(m)), [f"{c} (n={n[c]})" for c in countries] + ["mean"])
        ax.set_title(ptitle, loc="left", pad=28)
        ax.grid(False)
    fig.suptitle(title, x=0.0, ha="left", fontsize=10, color=INK)
    fig.text(0.0, -0.01, "colour: blue = better than default TabICL, red = worse (capped at ±0.05 NRMSE, ±0.15 r); "
             "means over 3 pretraining seeds; n = test region-years (raw WOFOST: no DE 2021, WOFOST ends 2020)",
             fontsize=7.5, color=INK2)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)


def all_models():
    """Every model on the benchmark's test rows; runs that don't exist yet are left out."""
    runs_present = tuple((e, t) for e, t in ALL_RUNS if (BENCH / e / "per_run.csv").exists())
    tags = {t for _, t in runs_present}
    settings = {k: v for k, v in ALL.items() if k[0] is None or k[0] in tags}
    r = runs(runs_present)
    single = "18 European maize countries, single pass\n(large marker = mean over 3 pretraining seeds, small = each seed)"
    overview(r, settings, "results_all_models_overview", f"All models on the same test rows: {single}")
    heatmap(r, settings, "results_all_models_heatmap", "All models per country, walk-forward test years")
    print(f"wrote {OUT}/results_all_models_*.png")


if __name__ == "__main__":
    main()
    decisive_test()
    all_models()
