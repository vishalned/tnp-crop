"""Figures of the EU grid WOFOST dataset (scaling_main): where it is, and how its yields compare with CY-Bench.

    uv run python -m analysis.scaling.plot_data      # reads a 1 GB parquet: run it in a job

Writes to `analysis/scaling/figures/`:

- `data_map.png`: every location, coloured by its mean simulated yield; failed locations as crosses;
- `data_yield_by_country.png`: per country, the simulated yields (matured seasons) next to CY-Bench's
  reported yields (all region-years 2005-2020 with yield > 0, raw yield file, no outlier filtering);
- `data_year_to_year.png`: per country, the yearly national yield as % deviation from its 2005-2020
  mean, simulated (mean over locations) vs CY-Bench (production / harvested area), with their correlation r;

and `analysis/scaling/data_per_country.csv` with the numbers behind them.
"""

import os

import matplotlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

RUN = "/projects/prjs1788/tnp-crop/wofost_eu_grid"
TABLE = f"{RUN}/processed/wofost_maize_daily.parquet"
LOCATIONS = "data/raw/locations/locations_maize_eu27_cell01_n2000_s42.csv"
CYBENCH = "/projects/prjs1788/CropBench/data/cybench/cybench-data/maize"
OUT = "analysis/scaling"
YEARS = (2005, 2020)  # harvest years
CODE = {"Austria": "AT", "Belgium": "BE", "Bulgaria": "BG", "Croatia": "HR", "Czechia": "CZ", "Denmark": "DK",
        "France": "FR", "Germany": "DE", "Greece": "EL", "Hungary": "HU", "Italy": "IT", "Latvia": "LV",
        "Lithuania": "LT", "Netherlands": "NL", "Poland": "PL", "Portugal": "PT", "Romania": "RO", "Slovakia": "SK",
        "Spain": "ES", "Sweden": "SE"}
WOFOST, CYB = "#2a78d6", "#8a8984"


def load():
    sim = pq.read_table(TABLE, columns=["location_index", "country", "year", "yield_t_per_ha", "reached_maturity"]
                        ).to_pandas()
    sim = sim[sim["reached_maturity"].astype(bool) & sim["year"].between(*YEARS)]
    frames = []
    for country, code in CODE.items():
        path = f"{CYBENCH}/{code}/yield_maize_{code}.csv"
        if os.path.exists(path):
            frames.append(pd.read_csv(path).assign(country=country))
    cyb = pd.concat(frames)
    cyb = cyb[cyb["harvest_year"].between(*YEARS) & (cyb["yield"] > 0)]  # LV has only 0 placeholder rows
    return sim, cyb


def _spines(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def data_map(sim):
    locations = pd.read_csv(LOCATIONS)
    manifest = pd.read_csv(f"{RUN}/dataset_manifest.csv", usecols=["location_index", "status"])
    failed = locations.loc[sorted(set(manifest.loc[manifest["status"] != "success", "location_index"]))]
    mean = sim.groupby("location_index")["yield_t_per_ha"].mean()
    ok = locations.loc[mean.index]
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(ok["longitude"], ok["latitude"], c=mean, s=7, vmin=0,
                    cmap=LinearSegmentedColormap.from_list("blues", plt.get_cmap("Blues")(np.linspace(0.25, 1, 256))), linewidths=0)
    ax.scatter(failed["longitude"], failed["latitude"], marker="x", s=18, color="#1f1e1c", linewidths=0.8,
               label=f"failed ({len(failed)}: coast outside ERA5-Land or no SoilGrids data)")
    ax.set_xlim(-10, 30)
    ax.set_ylim(35, 61)
    ax.set_aspect(1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    fig.colorbar(sc, ax=ax, shrink=0.6, label="mean simulated yield 2005-2020 (t/ha)")
    ax.set_title(f"EU grid WOFOST dataset: {len(ok):,} locations (one per 0.1° maize cell)", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figures/data_map.png", dpi=170)
    plt.close(fig)


def per_country(sim, cyb):
    rows = []
    for country in CODE:
        s, c = sim[sim["country"] == country], cyb[cyb["country"] == country]
        if s.empty:
            continue
        s_year = s.groupby("year")["yield_t_per_ha"].mean()
        c_year = (c.groupby("harvest_year")["production"].sum() / c.groupby("harvest_year")["harvest_area"].sum()
                  if c["production"].notna().any() else c.groupby("harvest_year")["yield"].mean())
        both = pd.concat([s_year.rename("sim"), c_year.rename("cyb")], axis=1).dropna()
        rows.append({"country": country, "code": CODE[country], "locations": s["location_index"].nunique(),
                     "seasons": len(s), "sim_mean": s["yield_t_per_ha"].mean(), "sim_sd": s["yield_t_per_ha"].std(),
                     "cybench_regions": c["adm_id"].nunique(), "cybench_region_years": len(c),
                     "cybench_mean": c["yield"].mean() if len(c) else np.nan,
                     "cybench_sd": c["yield"].std() if len(c) else np.nan,
                     "year_to_year_r": both["sim"].corr(both["cyb"]) if len(both) > 3 else np.nan,
                     "years_compared": len(both)})
    return pd.DataFrame(rows).sort_values("locations", ascending=False)


def yield_by_country(sim, cyb, table):
    countries = list(table["country"])
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(countries) + 1.2))
    for i, country in enumerate(countries):
        for k, (data, color) in enumerate(((sim.loc[sim["country"] == country, "yield_t_per_ha"], WOFOST),
                                           (cyb.loc[cyb["country"] == country, "yield"], CYB))):
            if data.empty:
                continue
            b = ax.boxplot([data.to_numpy()], positions=[i + (k - 0.5) * 0.36], vert=False, widths=0.3,
                           patch_artist=True, showfliers=False, medianprops={"color": "white", "lw": 1.5},
                           whiskerprops={"color": color}, capprops={"color": color})
            b["boxes"][0].set(facecolor=color, edgecolor=color)
    row = table.set_index("country")
    ax.set_yticks(range(len(countries)),
                  [f"{c} ({row.loc[c, 'locations']} loc. / {row.loc[c, 'cybench_regions']} reg.)" for c in countries])
    ax.set_ylim(len(countries) - 0.5, -0.5)
    ax.set_xlabel("yield (t/ha), 2005-2020; box = p25-p75, whiskers 1.5 IQR")
    ax.plot([], [], color=WOFOST, lw=6, label="WOFOST, simulated (matured seasons)")
    ax.plot([], [], color=CYB, lw=6, label="CY-Bench, reported (region-years)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, frameon=False, fontsize=8)
    fig.suptitle("simulated vs reported maize yield per country", x=0.01, ha="left", fontsize=10)
    _spines(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figures/data_yield_by_country.png", dpi=170)
    plt.close(fig)


def year_to_year(sim, cyb, table):
    countries = list(table["country"])
    n_cols = 5
    n_rows = int(np.ceil(len(countries) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 2.3 * n_rows), sharex=True, sharey=True)
    row = table.set_index("country")
    for ax, country in zip(axes.flat, countries):
        s = sim[sim["country"] == country].groupby("year")["yield_t_per_ha"].mean()
        c = cyb[cyb["country"] == country]
        if c["production"].notna().any():
            c = c.groupby("harvest_year")["production"].sum() / c.groupby("harvest_year")["harvest_area"].sum()
        else:
            c = c.groupby("harvest_year")["yield"].mean()
        ax.axhline(0, color="#d6d4cf", lw=0.8)
        ax.plot(s.index, 100 * (s / s.mean() - 1), color=WOFOST, lw=2, label="WOFOST")
        if len(c):
            ax.plot(c.index, 100 * (c / c.mean() - 1), color=CYB, lw=2, label="CY-Bench")
        r = row.loc[country, "year_to_year_r"]
        ax.set_title(f"{country}  r = {r:.2f}" if np.isfinite(r) else f"{country}  (no CY-Bench yields)",
                     loc="left", fontsize=9)
        _spines(ax)
    for ax in axes.flat[len(countries):]:
        ax.set_visible(False)
    axes.flat[0].legend(frameon=False, fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("% from mean")
    fig.suptitle("year-to-year national yield: simulated (mean over locations) vs reported "
                 "(production / area); r = correlation of the two series", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figures/data_year_to_year.png", dpi=170)
    plt.close(fig)


def main():
    os.makedirs(f"{OUT}/figures", exist_ok=True)
    sim, cyb = load()
    table = per_country(sim, cyb)
    table.to_csv(f"{OUT}/data_per_country.csv", index=False, float_format="%.3f")
    print(table.to_string(index=False, float_format="%.2f"))
    data_map(sim)
    yield_by_country(sim, cyb, table)
    year_to_year(sim, cyb, table)
    print(f"wrote {OUT}/figures/")


if __name__ == "__main__":
    main()
