"""Data plots of the observational-noise experiment (nanotabicl/obs_noise.py).

    uv run python -m experiments.obs_noise.plot_noise      # reads a subset of the OM-fix table

`figures/noise_series.png`: one season's daily fPAR (dekads), surface and root-zone soil moisture, clean and with the noise of three tables
(small, medium and large sigma). `figures/noise_features.png`: the CY-Bench features the noise acts on (fpar_mean, cum_fpar_max, ssm_mean,
rsm_mean, pooled over the time blocks), clean vs noisy, on a few hundred random locations, noise drawn per table as in training.
`figures/noise_params.png`: sigma and bg drawn in the training run (its `noise_params.csv`), which should be uniform.
"""

import os

import matplotlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from nanotabicl.cybench_features import apply_normalizer, cybench_features, fit_normalizer, variable_of  # noqa: E402
from nanotabicl.obs_noise import ObservationNoise  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE = "/projects/prjs1788/tnp-crop/wofost_eu_grid_dense_omfix/processed/wofost_maize_daily.parquet"
RUN_DIR = "/projects/prjs1788/tnp-crop/nanotabicl/exp_4_obs_noise/wofost_seed0"
GREY, GREEN = "#7a7a7a", "#1b9e77"
SHOWN = ("fpar_mean", "cum_fpar_max", "ssm_mean", "rsm_mean")


def main():
    meta = pq.read_table(TABLE, columns=["location_index"]).to_pandas()
    locs = np.random.default_rng(0).choice(meta.location_index.unique(), 300, replace=False).tolist()
    df = pq.read_table(TABLE, filters=[("location_index", "in", locs)]).to_pandas()
    df = df[df["reached_maturity"].astype(bool)].reset_index(drop=True)
    raw, names = cybench_features(df, drop_sparse=False)
    normalizer = fit_normalizer(raw, names, rows=df["sowing_year"].between(2005, 2016).to_numpy())
    clean = np.nan_to_num(apply_normalizer(raw, names, normalizer), nan=0.0)
    noise = ObservationNoise(df, names, normalizer, seed=0)

    # --- one season, clean vs three noise draws
    season = int(df.index[(df.country == "Spain") & (df.sowing_year == 2012)][0]) if ((df.country == "Spain") & (df.sowing_year == 2012)).any() else 0
    params = {"small": np.array([0.05, 0.06, 0.05, 0.05]), "medium": np.array([0.15, 0.12, 0.15, 0.15]), "large": np.array([0.3, 0.2, 0.3, 0.3])}
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    days = np.arange(noise.series["fpar"].shape[1])
    for ax, var, label in zip(axes, ("fpar", "ssm", "rsm"), ("fPAR, %", "ssm, kg/m2 (0-10 cm)", "rsm, kg/m2 (0-100 cm)")):
        obs = noise.observed["dekad" if var == "fpar" else "daily"][season]
        ax.plot(days[obs], noise.series[var][season][obs], color="k", lw=2 if var != "fpar" else 0, marker="o" if var == "fpar" else None, ms=4, label="clean (WOFOST)")
        for (tag, p), color in zip(params.items(), ("#9ad0c1", "#1b9e77", "#005f46")):
            s = noise.noisy_series(np.array([season]), p, rng=np.random.default_rng(1))[var][0]
            ax.plot(days[obs], s[obs], color=color, lw=0.9, marker="o" if var == "fpar" else None, ms=3, alpha=0.9,
                    label=f"noisy, sigma {p[0] if var == 'fpar' else p[2 if var == 'rsm' else 2]:.2f}" + (f", bg {p[1]:.2f}" if var == "fpar" else ""))
        ax.set_ylabel(label)
        ax.legend(fontsize=7, frameon=False, ncol=4)
    axes[-1].set_xlabel("day of the simulation window")
    fig.suptitle(f"Clean vs noisy daily series, one season ({df.country[season]}, sowing year {df.sowing_year[season]})")
    fig.tight_layout()
    fig.savefig(f"{HERE}/figures/noise_series.png", dpi=140)
    plt.close(fig)

    # --- features, clean vs noisy: tables of ~600 rows, params drawn per table
    rng_rows = np.random.default_rng(1)
    noisy = clean.copy()
    for _ in range(40):
        rows = np.sort(rng_rows.choice(len(df), 600, replace=False))
        for col, v in noise.apply(rows, noise.draw()).items():
            noisy[rows, col] = v
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    for ax, var in zip(axes, SHOWN):
        cols = [j for j, n in enumerate(names) if variable_of(n) == var]
        a, b = clean[:, cols].ravel(), noisy[:, cols].ravel()
        lo, hi = np.percentile(np.concatenate([a, b]), [0.5, 99.5])
        bins = np.linspace(lo, hi, 60)
        ax.hist(a, bins, density=True, alpha=0.55, color=GREY, label="clean")
        ax.hist(b, bins, density=True, alpha=0.55, color=GREEN, label="noisy (as in training)")
        ax.set_title(f"{var} (normalised, all blocks)", fontsize=9)
        ax.set_yticks([])
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{HERE}/figures/noise_features.png", dpi=140)
    plt.close(fig)
    changed = (np.abs(noisy - clean) > 1e-6).any(axis=0)
    print(f"columns that differ between clean and noisy: {int(changed.sum())} of {len(names)}: {sorted({variable_of(n) for n, c in zip(names, changed) if c})}")

    # --- the drawn parameters of the training run
    path = f"{RUN_DIR}/noise_params.csv"
    if os.path.exists(path):
        p = pd.read_csv(path)
        fig, axes = plt.subplots(1, 4, figsize=(14, 3))
        for ax, col, rng_ in zip(axes, ("sigma_fpar", "bg_fpar", "sigma_ssm", "sigma_rsm"), ((0, 0.3), (0.05, 0.2), (0, 0.3), (0, 0.3))):
            ax.hist(p[col], bins=30, range=rng_, color=GREEN)
            ax.set_title(f"{col} ({len(p)} tables)", fontsize=9)
        fig.tight_layout()
        fig.savefig(f"{HERE}/figures/noise_params.png", dpi=140)
        print(p.describe().round(3).to_string())


if __name__ == "__main__":
    main()
