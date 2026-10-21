"""Compare the WOFOST feature table with CY-Bench's, built by the benchmark itself.

Builds a CY-Bench country's tabular feature table with AgML-CY-Bench-Modeling's
own `DataFactory` and its real configs (`feature_design.yaml`,
`geometric_transform.yaml`) with `exclude_features=[vpd, ndvi, drainage_class]`
(what WOFOST doesn't have; the benchmark's `tabicl-wofost` branch), without
normalisation (raw units), and compares it with the WOFOST table from
`cybench_features`:

    python -m nanotabicl.cybench_check --table /projects/prjs1788/tnp-crop/processed/wofost_maize_daily.parquet \
        --cybench-data /projects/prjs1788/CropBench/data/cybench/cybench-data --country DE --out docs/figures

Prints whether the column names (in order) are equal, writes per-feature
statistics (`cybench_vs_wofost_<crop>_<CC>.csv`, with flags for suspicious
differences) and distribution plots (`cybench_vs_wofost_<crop>_<CC>_<source>.png`).
Needs `cybench` (`uv pip install -e ../AgML-CY-Bench-Modeling --no-deps`).
Loading a country's daily CSVs takes a few minutes: run it in a job.
"""

import argparse
import os

import numpy as np
import pandas as pd
import yaml

from nanotabicl.cybench_features import STATIC_FEATURES, cybench_features, variable_of

# what WOFOST doesn't have, left out with the benchmark's `dataset.exclude_features` option
EXCLUDE_FEATURES = ["vpd", "ndvi", "drainage_class"]


def benchmark_feature_table(cybench_data: str, crop: str, country: str, horizon: str = "eos",
                            benchmark_dir: str = None, filter_samples=None, return_y: bool = False):
    """CY-Bench's raw feature table for one crop/country, built by the benchmark's DataFactory
    (and its yields, `return_y`; `filter_samples=["flag_yield_outlier"]` drops flagged yields as the
    benchmark runs do, needs the yield_quality files)."""
    from omegaconf import OmegaConf

    import cybench
    from cybench.datasets import data_factory

    benchmark_dir = benchmark_dir or os.path.dirname(os.path.dirname(os.path.abspath(cybench.__file__)))
    conf = os.path.join(benchmark_dir, "cybench", "conf", "dataset")
    load = lambda *parts: yaml.safe_load(open(os.path.join(conf, *parts)))  # noqa: E731
    temporal = load("temporal", "feature_design.yaml")
    temporal["season"]["end_of_sequence"] = horizon
    cfg = OmegaConf.create({
        "crop": load("crop", f"{crop}.yaml"), "country": country, "name": f"{crop}_{country}",
        "min_year": 2000, "max_year": 2024, "framework": "pandas", "use_cache": False,
        # the benchmark's pure-pandas alignment (same result); its numpy path writes into arrays that
        # pandas 3 (this repo's env) returns read-only. The benchmark's own env has pandas 2.
        "use_memory_optimization": False,
        "temporal": temporal, "non_temporal": load("non_temporal", "geometric_transform.yaml"),
        "exclude_features": EXCLUDE_FEATURES,  # needs the benchmark's tabicl-wofost branch
        "target": {"name": "yield", "filter_samples": filter_samples},  # filtering doesn't change the columns
    })
    data_factory.PATH_DATA_DIR = cybench_data
    dataset = data_factory.DataFactory(cfg).build()
    return (dataset.x, dataset.y["yield"]) if return_y else dataset.x


def compare(wofost: pd.DataFrame, cybench: pd.DataFrame) -> pd.DataFrame:
    """Per-feature statistics of both tables and flags for suspicious differences."""
    rows = []
    for name in cybench.columns:
        a = cybench[name].to_numpy(float)
        b = wofost[name].to_numpy(float) if name in wofost else np.full(1, np.nan)
        stats = {"feature": name}
        for tag, v in (("cybench", a), ("wofost", b)):
            v = v[~np.isnan(v)]
            stats.update({f"{tag}_mean": v.mean() if len(v) else np.nan, f"{tag}_std": v.std() if len(v) else np.nan,
                          f"{tag}_p01": np.percentile(v, 1) if len(v) else np.nan,
                          f"{tag}_p99": np.percentile(v, 99) if len(v) else np.nan,
                          f"{tag}_zero_frac": float((v == 0).mean()) if len(v) else np.nan})
        scale = max(stats["cybench_std"], 1e-9)
        stats["mean_shift_in_cybench_std"] = (stats["wofost_mean"] - stats["cybench_mean"]) / scale
        stats["std_ratio"] = stats["wofost_std"] / scale
        flags = []
        if abs(stats["mean_shift_in_cybench_std"]) > 2:
            flags.append("mean shift > 2 std")
        if not 0.33 < stats["std_ratio"] < 3:
            flags.append("spread x3 different")
        if abs(stats["wofost_zero_frac"] - stats["cybench_zero_frac"]) > 0.2:
            flags.append("zeros")
        stats["flags"] = "; ".join(flags)
        rows.append(stats)
    return pd.DataFrame(rows)


def plot(wofost: pd.DataFrame, cybench: pd.DataFrame, names: list, path: str, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_cols = 9
    n_rows = int(np.ceil(len(names) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 1.7 * n_rows), squeeze=False)
    for ax, name in zip(axes.ravel(), names):
        a, b = cybench[name].dropna().to_numpy(float), wofost[name].dropna().to_numpy(float)
        lo, hi = np.percentile(np.concatenate([a, b]), [0.5, 99.5])
        bins = np.linspace(lo, hi if hi > lo else lo + 1, 40)
        ax.hist(a, bins=bins, density=True, alpha=0.5, color="tab:blue", label="CY-Bench")
        ax.hist(b, bins=bins, density=True, alpha=0.5, color="tab:orange", label="WOFOST")
        ax.set_title(name, fontsize=7)
        ax.tick_params(labelsize=5)
        ax.set_yticks([])
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=6)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--table", required=True, help="processed WOFOST table (--align calendar)")
    p.add_argument("--cybench-data", required=True, help="CY-Bench data dir (<crop>/<CC>/*.csv)")
    p.add_argument("--country", default="DE")
    p.add_argument("--horizon", default="eos")
    p.add_argument("--out", default="docs/figures")
    args = p.parse_args()

    df = pd.read_parquet(args.table)
    crop = df["crop"].iloc[0]
    x, names = cybench_features(df, horizon=args.horizon)
    wofost = pd.DataFrame(x, columns=names)
    cyb = benchmark_feature_table(args.cybench_data, crop, args.country, args.horizon)

    same = names == list(cyb.columns)
    print(f"WOFOST: {len(names)} features, CY-Bench {crop} {args.country}: {cyb.shape[1]} features "
          f"({len(cyb)} rows); names equal in order: {same}")
    if not same:
        print("only WOFOST:", [n for n in names if n not in cyb.columns])
        print("only CY-Bench:", [n for n in cyb.columns if n not in names])
        print("first difference:", next(((i, a, b) for i, (a, b) in enumerate(zip(names, cyb.columns)) if a != b), None))

    os.makedirs(args.out, exist_ok=True)
    stem = os.path.join(args.out, f"cybench_vs_wofost_{crop}_{args.country}_{args.horizon}")
    stats = compare(wofost, cyb)
    stats.to_csv(f"{stem}.csv", index=False)
    flagged = stats[stats["flags"] != ""]
    print(f"{len(flagged)}/{len(stats)} features flagged; per variable:")
    flagged = flagged.assign(variable=flagged["feature"].map(variable_of))
    print(flagged.groupby("variable")["flags"].agg(lambda f: sorted(set("; ".join(f).split("; ")))).to_string())

    groups = {"meteo": [n for n in names if n not in STATIC_FEATURES and not variable_of(n).startswith(("fpar", "cum_fpar", "ssm", "rsm"))],
              "fpar_soil_moisture": [n for n in names if variable_of(n).startswith(("fpar", "cum_fpar", "ssm", "rsm"))],
              "static": [n for n in names if n in STATIC_FEATURES]}
    for group, cols in groups.items():
        cols = [c for c in cols if c in cyb.columns]
        plot(wofost, cyb, cols, f"{stem}_{group}.png", f"{crop} {args.country} ({args.horizon}): {group}, CY-Bench vs WOFOST")
    print(f"wrote {stem}.csv and {stem}_*.png")


if __name__ == "__main__":
    main()
