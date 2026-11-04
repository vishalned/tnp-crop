"""Cultivar variation (TSUM1 and TSUM2 x U(0.8, 1.2) per location): what changed in the data, against the OM-fix table it is built from.

    uv run python -m analysis.cultivar_tsum.compare_data [--new TABLE] [--max-location N]     # run through SLURM (the tables are 5-6 GB)

Both tables are the same locations, years and sowing dates (seasons match one to one on location, sowing year and jitter); only the temperature
sums differ. 1) the usable filter (a location is usable if it has a matured season with a finite yield in every one of the 12 training years,
2005-2016, as in densify_grid_locations.py): usable locations and share of matured seasons per country; 2) diversity: season length, yield,
spread over locations, year-to-year spread within a location, and the yield's dependence on the factor. Writes into this folder:
`usable_per_country.csv`, `diversity_per_country.csv`, `summary.txt` and `figures/{usable,diversity,factor}.png`.
"""

import argparse
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
OLD = "/projects/prjs1788/tnp-crop/wofost_eu_grid_dense_omfix/processed/wofost_maize_daily.parquet"
NEW = "/projects/prjs1788/tnp-crop/wofost_eu_grid_dense_tsum/processed/wofost_maize_daily.parquet"
TRAIN = (2005, 2016)
COLS = ["location_index", "country", "sowing_year", "jitter_index", "yield_t_per_ha", "reached_maturity", "season_length_days"]
GREY, GREEN = "#7a7a7a", "#1b9e77"


def load(path, max_location, extra=()):
    d = pd.read_parquet(path, columns=COLS + list(extra), filters=[("location_index", "<", max_location)] if max_location else None)
    d["matured"] = d["reached_maturity"].astype(bool) & np.isfinite(d["yield_t_per_ha"])
    return d


def usable(d):
    """Series location_index -> usable (a matured season with a finite yield in each of the 12 training years)."""
    t = d[d.matured & d.sowing_year.between(*TRAIN)]
    years = t.groupby("location_index").sowing_year.nunique()
    full = pd.Series(False, index=d.location_index.unique())
    full.loc[years.index[years == TRAIN[1] - TRAIN[0] + 1]] = True
    return full


def diversity(d, full):
    """Per country, on matured training-year seasons of the usable locations (the data the tables are built from)."""
    t = d[d.matured & d.sowing_year.between(*TRAIN) & d.location_index.map(full).fillna(False)]
    loc_year = t.groupby(["country", "location_index", "sowing_year"]).yield_t_per_ha.mean()
    loc_mean = loc_year.groupby(["country", "location_index"]).mean()
    return pd.DataFrame({
        "yield_mean": t.groupby("country").yield_t_per_ha.mean(),
        "yield_sd": t.groupby("country").yield_t_per_ha.std(),
        "season_length_mean": t.groupby("country").season_length_days.mean(),
        "season_length_sd": t.groupby("country").season_length_days.std(),
        "spatial_sd": loc_mean.groupby("country").std(),  # spread of the locations' mean yields
        "interannual_sd": loc_year.groupby(["country", "location_index"]).std().groupby("country").mean(),  # year to year, within a location
    })


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--old", default=OLD)
    p.add_argument("--new", default=NEW)
    p.add_argument("--max-location", type=int, default=None, help="only the locations below this index (test)")
    p.add_argument("--out", default=OUT)
    a = p.parse_args()
    os.makedirs(f"{a.out}/figures", exist_ok=True)
    old, new = load(a.old, a.max_location), load(a.new, a.max_location, extra=["tsum_factor"])
    f_old, f_new = usable(old), usable(new)
    country = new.drop_duplicates("location_index").set_index("location_index").country
    factor = new.drop_duplicates("location_index").set_index("location_index").tsum_factor

    rows = pd.DataFrame({
        "locations": country.value_counts(),
        "usable_old": f_old.groupby(country).sum(), "usable_new": f_new.groupby(country).sum(),
        "matured_share_old": old.groupby("country").matured.mean(), "matured_share_new": new.groupby("country").matured.mean(),
    }).sort_values("locations", ascending=False)
    rows.loc["ALL"] = [rows.locations.sum(), rows.usable_old.sum(), rows.usable_new.sum(), old.matured.mean(), new.matured.mean()]
    rows.round(4).to_csv(f"{a.out}/usable_per_country.csv")

    div_old, div_new = diversity(old, f_old), diversity(new, f_new)
    div = div_old.join(div_new, lsuffix="_old", rsuffix="_new")
    div.round(4).to_csv(f"{a.out}/diversity_per_country.csv")

    # usable in the new table only / old table only
    both = pd.concat([f_old.rename("old"), f_new.rename("new")], axis=1)
    pooled = lambda d, c: d[c].dropna()
    lines = [f"tsum factor: n={len(factor)}, mean {factor.mean():.3f}, min {factor.min():.3f}, max {factor.max():.3f}",
             f"seasons: {len(new):,}; matured with finite yield: old {old.matured.mean():.3%}, new {new.matured.mean():.3%}",
             f"usable locations: old {int(f_old.sum()):,}, new {int(f_new.sum()):,} of {len(f_new):,}  "
             f"(usable in both {int((both.old & both.new).sum()):,}, only old {int((both.old & ~both.new).sum()):,}, only new {int((~both.old & both.new).sum()):,})",
             f"countries with >= 682 usable locations (4096 rows / 6 rows per point): old {(rows.usable_old.drop('ALL') >= 682).sum()}, new {(rows.usable_new.drop('ALL') >= 682).sum()}",
             "", "pooled over matured training-year seasons of the usable locations:"]
    for name, d, full in (("old", old, f_old), ("new", new, f_new)):
        t = d[d.matured & d.sowing_year.between(*TRAIN) & d.location_index.map(full).fillna(False)]
        lines.append(f"  {name}: yield mean {t.yield_t_per_ha.mean():.2f} sd {t.yield_t_per_ha.std():.2f} t/ha; "
                     f"season length mean {t.season_length_days.mean():.1f} sd {t.season_length_days.std():.1f} d")
    lines += ["", "mean of the per-country values (diversity):", div.mean().round(3).to_string()]
    open(f"{a.out}/summary.txt", "w").write("\n".join(lines))
    print("\n".join(lines))
    print(rows.round(3).to_string())

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    # 1) usable filter
    r = rows.drop("ALL")
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
    x, w = np.arange(len(r)), 0.4
    axes[0].bar(x - w / 2, r.usable_old, w, color=GREY, label="OM-fix table (original temperature sums)")
    axes[0].bar(x + w / 2, r.usable_new, w, color=GREEN, label="+ cultivar variation (TSUM x U(0.8, 1.2))")
    axes[0].plot(x, r.locations, "k_", ms=12, label="locations simulated")
    axes[1].bar(x - w / 2, r.matured_share_old * 100, w, color=GREY)
    axes[1].bar(x + w / 2, r.matured_share_new * 100, w, color=GREEN)
    for ax, t in zip(axes, ("usable locations (matured in all 12 training years)", "seasons that matured, %")):
        ax.set_xticks(x, r.index, rotation=60, ha="right")
        ax.set_title(t)
        ax.grid(axis="y", color="#e5e5e5", lw=0.6)
        ax.set_axisbelow(True)
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{a.out}/figures/usable.png", dpi=150)
    plt.close(fig)
    # 2) diversity
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    tr = lambda d, full: d[d.matured & d.sowing_year.between(*TRAIN) & d.location_index.map(full).fillna(False)]
    t_old, t_new = tr(old, f_old), tr(new, f_new)
    for ax, col, label, bins in ((axes[0, 0], "season_length_days", "season length, days", 60), (axes[0, 1], "yield_t_per_ha", "yield, t/ha", 60)):
        ax.hist(t_old[col], bins=bins, density=True, alpha=0.6, color=GREY, label="OM-fix table")
        ax.hist(t_new[col], bins=bins, density=True, alpha=0.6, color=GREEN, label="+ cultivar variation")
        ax.set_xlabel(label)
        ax.set_ylabel("density")
    axes[0, 0].legend(fontsize=8, frameon=False)
    d = div.reindex(r.index)
    for ax, col, title in ((axes[0, 2], "yield_sd", "yield sd within a country, t/ha"), (axes[1, 0], "spatial_sd", "sd over locations of the mean yield, t/ha"),
                           (axes[1, 1], "interannual_sd", "year-to-year sd within a location, t/ha")):
        ax.bar(x - w / 2, d[f"{col}_old"], w, color=GREY)
        ax.bar(x + w / 2, d[f"{col}_new"], w, color=GREEN)
        ax.set_xticks(x, r.index, rotation=60, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", color="#e5e5e5", lw=0.6)
        ax.set_axisbelow(True)
    axes[1, 2].bar(x - w / 2, d.season_length_sd_old, w, color=GREY)
    axes[1, 2].bar(x + w / 2, d.season_length_sd_new, w, color=GREEN)
    axes[1, 2].set_xticks(x, r.index, rotation=60, ha="right")
    axes[1, 2].set_title("season length sd within a country, days")
    fig.suptitle("Data diversity: matured training-year seasons of the usable locations")
    fig.tight_layout()
    fig.savefig(f"{a.out}/figures/diversity.png", dpi=150)
    plt.close(fig)
    # 3) the factor's effect
    n = new[new.matured].assign(factor=lambda d: d.tsum_factor)
    n["bin"] = pd.cut(n.factor, np.linspace(0.8, 1.2, 9))
    old_m = old[old.matured].set_index(["location_index", "sowing_year", "jitter_index"]).yield_t_per_ha
    n = n.join(old_m.rename("yield_old"), on=["location_index", "sowing_year", "jitter_index"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    g = n.groupby("bin", observed=True)
    mid = [i.mid for i in g.size().index]
    axes[0].plot(mid, g.season_length_days.mean(), "o-", color=GREEN)
    axes[0].set_xlabel("TSUM factor")
    axes[0].set_ylabel("mean season length, days (matured seasons)")
    ratio = (n.yield_t_per_ha / n.yield_old).where(n.yield_old > 0).groupby(n.bin, observed=True).median()
    axes[1].plot(mid, ratio, "o-", color=GREEN)
    axes[1].axhline(1, color="k", lw=0.6)
    axes[1].set_xlabel("TSUM factor")
    axes[1].set_ylabel("median yield / yield with the original sums (same season)")
    for ax in axes:
        ax.grid(color="#e5e5e5", lw=0.6)
    fig.tight_layout()
    fig.savefig(f"{a.out}/figures/factor.png", dpi=150)


if __name__ == "__main__":
    main()
