"""Old vs new densified EU grid (before / after the organic-matter fix OM = SOC / 0.58): did the data change?

    uv run python -m analysis.soc_fix.compare_distributions      # reads two ~5.6 GB parquets: run it in a job

The two tables hold the same seasons (same locations, sowing years and jitters, same weather); only the soil differs.
Seasons are matched on (location_index, sowing_year, jitter_index). Writes to `analysis/soc_fix/`:

- `figures/soil_static.png`: awc and bulk_density per location (old vs new), per-country mean awc;
- `figures/yield.png`: yield distribution, per-season scatter old vs new, per-country mean yield;
- `figures/features.png`: CY-Bench's raw feature distributions (pooled over time blocks) on 500 random locations;
- `per_country.csv`, `feature_shift.csv`: the numbers behind them, incl. a check that weather features are identical.
"""

import os

import matplotlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from nanotabicl.cybench_features import cybench_features, variable_of  # noqa: E402

OLD = "/projects/prjs1788/tnp-crop/wofost_eu_grid_dense/processed/wofost_maize_daily.parquet"
NEW = "/projects/prjs1788/tnp-crop/wofost_eu_grid_dense_omfix/processed/wofost_maize_daily.parquet"
OUT = os.path.dirname(os.path.abspath(__file__))
KEY = ["location_index", "sowing_year", "jitter_index"]
SCALARS = KEY + ["country", "awc", "bulk_density", "yield_t_per_ha", "reached_maturity"]
OLD_C, NEW_C = "#7a7a7a", "#1b9e77"
WEATHER = ("tmin", "tmax", "tavg", "prec", "rad", "et0", "cwb", "gdd", "cum_gdd", "cum_prec", "cum_cwb")  # soil-independent (fpar, ssm, rsm are model outputs)


def scalars(path: str) -> pd.DataFrame:
    df = pq.read_table(path, columns=SCALARS).to_pandas()
    return df[df["reached_maturity"].astype(bool)].drop(columns="reached_maturity")


def hist2(ax, old, new, title, bins=60):
    lo, hi = np.nanpercentile(np.concatenate([old, new]), [0.5, 99.5])
    b = np.linspace(lo, hi if hi > lo else lo + 1, bins)
    ax.hist(old, b, density=True, alpha=0.55, color=OLD_C, label="old (OM = SOC x 0.58)")
    ax.hist(new, b, density=True, alpha=0.55, color=NEW_C, label="new (OM = SOC / 0.58)")
    ax.set_title(title, fontsize=9)
    ax.set_yticks([])


def main():
    old, new = scalars(OLD), scalars(NEW)
    m = old.merge(new, on=KEY, suffixes=("_old", "_new"))
    print(f"seasons: old {len(old)}, new {len(new)}, matched {len(m)}", flush=True)

    # --- soil statics, per location
    loc = m.groupby("location_index").agg(country=("country_old", "first"), awc_old=("awc_old", "first"), awc_new=("awc_new", "first"),
                                          bd_old=("bulk_density_old", "first"), bd_new=("bulk_density_new", "first"))
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    hist2(ax[0], loc.awc_old.to_numpy(), loc.awc_new.to_numpy(), "awc per location")
    hist2(ax[1], loc.bd_old.to_numpy(), loc.bd_new.to_numpy(), "bulk_density per location")
    c = loc.groupby("country")[["awc_old", "awc_new"]].mean().sort_values("awc_old")
    y = np.arange(len(c))
    ax[2].barh(y - 0.2, c.awc_old, 0.4, color=OLD_C, label="old")
    ax[2].barh(y + 0.2, c.awc_new, 0.4, color=NEW_C, label="new")
    ax[2].set_yticks(y, c.index, fontsize=7)
    ax[2].set_title("mean awc per country", fontsize=9)
    ax[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figures/soil_static.png", dpi=140)
    plt.close(fig)

    # --- yield
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    hist2(ax[0], m.yield_t_per_ha_old.to_numpy(), m.yield_t_per_ha_new.to_numpy(), "simulated yield, t/ha (matured seasons)")
    s = m.sample(min(len(m), 100_000), random_state=0)
    ax[1].hexbin(s.yield_t_per_ha_old, s.yield_t_per_ha_new, gridsize=60, bins="log", cmap="viridis", mincnt=1)
    lim = [0, float(np.nanpercentile(s.yield_t_per_ha_old, 99.9))]
    ax[1].plot(lim, lim, "w--", lw=0.8)
    ax[1].set_xlabel("old yield")
    ax[1].set_ylabel("new yield")
    ax[1].set_title("per season (100k sample)", fontsize=9)
    cy = m.groupby("country_old")[["yield_t_per_ha_old", "yield_t_per_ha_new"]].mean().sort_values("yield_t_per_ha_old")
    y = np.arange(len(cy))
    ax[2].barh(y - 0.2, cy.yield_t_per_ha_old, 0.4, color=OLD_C)
    ax[2].barh(y + 0.2, cy.yield_t_per_ha_new, 0.4, color=NEW_C)
    ax[2].set_yticks(y, cy.index, fontsize=7)
    ax[2].set_title("mean yield per country, t/ha", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figures/yield.png", dpi=140)
    plt.close(fig)

    per_country = loc.groupby("country").agg(n_locations=("awc_old", "size"), awc_old=("awc_old", "mean"), awc_new=("awc_new", "mean"))
    per_country = per_country.join(m.groupby("country_old").agg(n_seasons=("yield_t_per_ha_old", "size"), yield_old=("yield_t_per_ha_old", "mean"),
                                                                yield_new=("yield_t_per_ha_new", "mean")))
    per_country["awc_change_pct"] = 100 * (per_country.awc_new / per_country.awc_old - 1)
    per_country["yield_change_pct"] = 100 * (per_country.yield_new / per_country.yield_old - 1)
    per_country.round(4).to_csv(f"{OUT}/per_country.csv")
    print(per_country.round(3).to_string(), flush=True)

    # --- CY-Bench features on a random sample of locations (same seasons in both tables)
    sample = np.random.default_rng(0).choice(loc.index.to_numpy(), 500, replace=False).tolist()
    tables = {}
    for tag, path in (("old", OLD), ("new", NEW)):
        df = pq.read_table(path, filters=[("location_index", "in", sample)]).to_pandas()
        df = df[df["reached_maturity"].astype(bool)].sort_values(KEY).reset_index(drop=True)
        x, names = cybench_features(df)
        tables[tag] = (df[KEY], x, names)
    (k_old, x_old, names), (k_new, x_new, names_new) = tables["old"], tables["new"]
    assert names == names_new
    both = k_old.merge(k_new, on=KEY).shape[0]
    assert both == len(k_old) == len(k_new), "the sampled seasons differ between the tables"
    variables = list(dict.fromkeys(variable_of(n) for n in names))
    rows = []
    fig, axes = plt.subplots(5, 5, figsize=(17, 12))
    for ax, var in zip(axes.ravel(), variables):
        cols = [i for i, n in enumerate(names) if variable_of(n) == var]
        a, b = x_old[:, cols], x_new[:, cols]
        a_f, b_f = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a_f) and len(b_f):
            hist2(ax, a_f, b_f, var)
        rows.append({"feature": var, "mean_old": a_f.mean(), "mean_new": b_f.mean(), "std_old": a_f.std(),
                     "shift_in_old_std": (b_f.mean() - a_f.mean()) / max(a_f.std(), 1e-9),
                     "max_abs_diff_same_season": float(np.nanmax(np.abs(a - b))),
                     "weather_feature": var.rsplit("_", 1)[0] in WEATHER or var.split("_")[0] in WEATHER})
    for ax in axes.ravel()[len(variables):]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figures/features.png", dpi=110)
    plt.close(fig)
    shift = pd.DataFrame(rows)
    shift.round(6).to_csv(f"{OUT}/feature_shift.csv", index=False)
    print(shift.round(4).to_string(), flush=True)
    bad = shift[shift.weather_feature & (shift.max_abs_diff_same_season > 1e-4)]
    print("weather features that differ between the tables (should be none):", bad.feature.tolist(), flush=True)


if __name__ == "__main__":
    main()
