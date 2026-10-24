"""Distribution gap between the WOFOST tables and CY-Bench, per feature and for the yield.

    python -m nanotabicl.distribution_gap --table /projects/prjs1788/tnp-crop/processed/wofost_maize_daily.parquet \
        --cybench-data /projects/prjs1788/CropBench/data/cybench/cybench-data --countries DE FR BE \
        --out docs/figures/distribution_gap

Datasets: all WOFOST seasons (matured, as in training), WOFOST per country, and CY-Bench per country
built by the benchmark's own DataFactory (raw units, yield-outlier flags applied as in the benchmark
runs; `cybench_check.benchmark_feature_table`). Writes:

- `hist_wofost_all_vs_cybench.png`: every feature + yield, WOFOST (all) against each CY-Bench country;
- `hist_wofost_<CC>_vs_cybench_<CC>.png`: the same country on both sides (countries WOFOST covers);
- `gap_heatmap.png`: per comparison and feature, (a) the mean shift in CY-Bench standard deviations
  (raw units) and (b) a shape gap: the Wasserstein distance between the two distributions after
  standardising each with its own mean/std -- what's left after the per-column standardisation the
  model applies to every table;
- `yield_correlation.png`: each feature's correlation with the yield, WOFOST against CY-Bench;
- `gap_stats.csv`: all of it per comparison and feature.

Loading a CY-Bench country takes a few minutes: run it in a job (scripts/slurm/run_distribution_gap.sh).
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from nanotabicl.cybench_check import benchmark_feature_table
from nanotabicl.cybench_features import STATIC_FEATURES, cybench_features, variable_of

WOFOST_COUNTRIES = {  # locations CSV country name -> CY-Bench country code
    "Austria": "AT", "Belgium": "BE", "Bulgaria": "BG", "Croatia": "HR", "Czechia": "CZ", "Denmark": "DK",
    "France": "FR", "Germany": "DE", "Greece": "EL", "Hungary": "HU", "Italy": "IT", "Latvia": "LV",
    "Lithuania": "LT", "Netherlands": "NL", "Poland": "PL", "Portugal": "PT", "Romania": "RO", "Slovakia": "SK",
    "Slovenia": "SI", "Spain": "ES", "Sweden": "SE",
}


def load_wofost(path: str, horizon: str = "eos") -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "reached_maturity" in df:
        df = df[df["reached_maturity"].astype(bool)].reset_index(drop=True)
    x, names = cybench_features(df, horizon=horizon)
    out = pd.DataFrame(x, columns=names)
    out["yield"] = df["yield_t_per_ha"].to_numpy(float)
    out["country"] = df["country"].map(WOFOST_COUNTRIES).fillna(df["country"]).to_numpy()
    return out


def _clean(v) -> np.ndarray:
    v = np.asarray(v, float)
    return v[np.isfinite(v)]


def gap_stats(a: pd.DataFrame, b: pd.DataFrame, features: list) -> pd.DataFrame:
    """a = WOFOST side, b = CY-Bench side."""
    rows = []
    for f in features:
        x, y = _clean(a[f]), _clean(b[f])
        sx, sy = x.std(), y.std()
        row = {"feature": f, "wofost_mean": x.mean(), "cybench_mean": y.mean(), "wofost_std": sx, "cybench_std": sy,
               "mean_shift_in_cybench_std": (x.mean() - y.mean()) / max(sy, 1e-9),
               "std_ratio": sx / max(sy, 1e-9), "ks_raw": ks_2samp(x, y).statistic,
               "shape_gap": wasserstein_distance((x - x.mean()) / max(sx, 1e-9), (y - y.mean()) / max(sy, 1e-9)),
               "zero_frac_wofost": float((x == 0).mean()), "zero_frac_cybench": float((y == 0).mean())}
        if f != "yield":
            for tag, d in (("wofost", a), ("cybench", b)):
                pair = d[[f, "yield"]].dropna()
                row[f"corr_yield_{tag}"] = pair[f].corr(pair["yield"]) if pair[f].std() > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def hist_grid(datasets: dict, features: list, path: str, title: str, n_cols: int = 12) -> None:
    """One panel per feature; the first dataset filled grey, the others as coloured outlines."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["0.6", "tab:blue", "tab:red", "tab:green", "tab:purple"]
    n_rows = int(np.ceil(len(features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.95 * n_cols, 1.55 * n_rows), squeeze=False)
    for ax, f in zip(axes.ravel(), features):
        values = {k: _clean(d[f]) for k, d in datasets.items()}
        pooled = np.concatenate([v for v in values.values() if len(v)])
        lo, hi = np.percentile(pooled, [0.5, 99.5])
        bins = np.linspace(lo, hi if hi > lo else lo + 1, 40)
        for i, (k, v) in enumerate(values.items()):
            if i == 0:
                ax.hist(v, bins=bins, density=True, color=colors[i], alpha=0.6, label=k)
            else:
                ax.hist(v, bins=bins, density=True, histtype="step", lw=1.2, color=colors[i], label=k)
        ax.set_title(f, fontsize=7, fontweight="bold" if f == "yield" else None)
        ax.tick_params(labelsize=5)
        ax.set_yticks([])
    for ax in axes.ravel()[len(features):]:
        ax.axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=10, ncol=len(labels))
    fig.suptitle(title, fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(path, dpi=80)
    plt.close(fig)


def _matrix(stats: pd.DataFrame, column: str, features: list) -> tuple:
    """(variables x blocks) matrix of a stat; static features and yield as rows with one value (block 0)."""
    temporal = [f for f in features if f not in STATIC_FEATURES and f != "yield"]
    variables = list(dict.fromkeys(variable_of(f) for f in temporal))
    n_blocks = max(int(f.rsplit("_", 1)[1]) for f in temporal) + 1
    rows = variables + [f for f in features if f in STATIC_FEATURES] + ["yield"]
    m = np.full((len(rows), n_blocks), np.nan)
    s = stats.set_index("feature")[column]
    for i, r in enumerate(rows):
        if r in variables:
            for b in range(n_blocks):
                if f"{r}_{b}" in s:
                    m[i, b] = s[f"{r}_{b}"]
        elif r in s:
            m[i, 0] = s[r]
    return m, rows, n_blocks


def gap_heatmap(stats: dict, features: list, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [("mean_shift_in_cybench_std", "mean shift (CY-Bench std, raw units)", "RdBu_r", (-4, 4)),
               ("shape_gap", "shape gap (Wasserstein, each standardised)", "Reds", (0, 1))]
    fig, axes = plt.subplots(len(metrics), len(stats), figsize=(4.2 * len(stats), 16), squeeze=False)
    for j, (name, st) in enumerate(stats.items()):
        for i, (col, label, cmap, (vmin, vmax)) in enumerate(metrics):
            m, rows, n_blocks = _matrix(st, col, features)
            ax = axes[i, j]
            im = ax.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_xticks(range(n_blocks), [str(b) for b in range(n_blocks)], fontsize=7)
            ax.set_yticks(range(len(rows)), rows if j == 0 else [], fontsize=7)
            ax.set_xlabel("block (0 = last 30 days before the cutoff); statics/yield in column 0", fontsize=6)
            ax.set_title(f"{name}\n{label}", fontsize=8)
            for r in range(m.shape[0]):
                for b in range(m.shape[1]):
                    if np.isfinite(m[r, b]):
                        ax.text(b, r, f"{m[r, b]:.1f}", ha="center", va="center", fontsize=4.5)
            if j == len(stats) - 1:
                fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def yield_correlation_plot(stats: dict, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(stats), figsize=(4.6 * len(stats), 4.8), squeeze=False)
    groups = {"temperature/GDD": ("tmin", "tmax", "tavg", "gdd", "cum_gdd"), "water": ("prec", "cwb", "et0", "cum_prec", "cum_cwb"),
              "radiation": ("rad",), "fpar": ("fpar", "cum_fpar"), "soil moisture": ("ssm", "rsm")}
    colors = dict(zip(list(groups) + ["static"], ["tab:red", "tab:blue", "tab:orange", "tab:green", "tab:brown", "k"]))

    def group_of(f):
        if f in STATIC_FEATURES:
            return "static"
        var = variable_of(f)[: variable_of(f).rfind("_")]  # cum_gdd_max_3 -> cum_gdd_max -> cum_gdd
        return next((g for g, vs in groups.items() if var in vs), "static")

    for ax, (name, st) in zip(axes[0], stats.items()):
        st = st[st["feature"] != "yield"]
        for g in colors:
            sel = st[st["feature"].map(group_of) == g]
            ax.scatter(sel["corr_yield_cybench"], sel["corr_yield_wofost"], s=12, alpha=0.8, color=colors[g], label=g)
        ax.plot([-1, 1], [-1, 1], "k--", lw=0.8)
        ax.axhline(0, color="0.8", lw=0.6)
        ax.axvline(0, color="0.8", lw=0.6)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_xlabel("correlation with yield in CY-Bench", fontsize=8)
        ax.set_ylabel("correlation with yield in WOFOST", fontsize=8)
        both = st[["corr_yield_cybench", "corr_yield_wofost"]].dropna()
        agree = np.mean(np.sign(both["corr_yield_cybench"]) == np.sign(both["corr_yield_wofost"]))
        ax.set_title(f"{name}\nsame sign: {agree:.0%}, r = {both.corr().iloc[0, 1]:.2f}", fontsize=8)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--table", required=True)
    p.add_argument("--cybench-data", required=True)
    p.add_argument("--countries", nargs="+", default=["DE", "FR", "BE"])
    p.add_argument("--crop", default="maize")
    p.add_argument("--horizon", default="eos")
    p.add_argument("--out", default="docs/figures/distribution_gap")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    wofost = load_wofost(args.table, args.horizon)
    cybench = {}
    for cc in args.countries:
        x, y = benchmark_feature_table(args.cybench_data, args.crop, cc, args.horizon,
                                       filter_samples=["flag_yield_outlier"], return_y=True)
        cybench[cc] = x.assign(**{"yield": y.reindex(x.index).to_numpy(float)})
        print(f"CY-Bench {cc}: {len(x)} rows, {x.shape[1]} features")
    print(f"WOFOST: {len(wofost)} rows, countries {wofost['country'].value_counts().to_dict()}")

    features = [c for c in wofost.columns if c not in ("yield", "country")]
    for cc, d in cybench.items():
        missing = [f for f in features if f not in d.columns]
        if missing:
            print(f"  not in CY-Bench {cc}: {missing}")
        features = [f for f in features if f in d.columns]
    features += ["yield"]

    shown = dict(list(cybench.items())[:4])  # the overview has 4 outline colours; per-country plots below cover all
    hist_grid({"WOFOST (all)": wofost, **{f"CY-Bench {cc}": d for cc, d in shown.items()}}, features,
              os.path.join(args.out, "hist_wofost_all_vs_cybench.png"),
              f"{args.crop}, {args.horizon}: WOFOST (all simulated seasons) vs CY-Bench countries (raw units)")
    comparisons = {f"WOFOST all vs CY-Bench {cc}": (wofost, d) for cc, d in cybench.items()}
    for cc, d in cybench.items():
        w = wofost[wofost["country"] == cc]
        if len(w):
            hist_grid({f"WOFOST {cc}": w, f"CY-Bench {cc}": d}, features,
                      os.path.join(args.out, f"hist_wofost_{cc}_vs_cybench_{cc}.png"),
                      f"{args.crop}, {args.horizon}: WOFOST {cc} ({len(w)} seasons) vs CY-Bench {cc} ({len(d)} region-years)")
            comparisons[f"WOFOST {cc} vs CY-Bench {cc}"] = (w, d)

    stats = {name: gap_stats(a, b, features) for name, (a, b) in comparisons.items()}
    pd.concat([s.assign(comparison=name) for name, s in stats.items()]).to_csv(
        os.path.join(args.out, "gap_stats.csv"), index=False)
    gap_heatmap(stats, features, os.path.join(args.out, "gap_heatmap.png"))
    yield_correlation_plot(stats, os.path.join(args.out, "yield_correlation.png"))
    for name, st in stats.items():
        y = st[st["feature"] == "yield"].iloc[0]
        print(f"{name}: yield mean {y.wofost_mean:.2f} vs {y.cybench_mean:.2f} t/ha, std {y.wofost_std:.2f} vs "
              f"{y.cybench_std:.2f}; median |mean shift| {st['mean_shift_in_cybench_std'].abs().median():.2f} std, "
              f"median shape gap {st['shape_gap'].median():.2f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
