"""Data plots of the irrigation-map experiment.

    uv run python -m experiments.irrigation_map.plot_irrigation      # reads three parquets (scalar columns only)

`figures/irrigation_map.png`: (left) where the locations are, coloured by what the MIRCA-OS maize map says (irrigated draw / rainfed /
no maize area in the cell); (right) per country the share of irrigated locations, with the map's mean irrigated share.
`figures/irrigation_yields.png`: per country the mean simulated yield of the rainfed run, the fully irrigated run (PP) and the
map mix, against CY-Bench's reported mean; and the correlation of the national year-to-year yield with the reported one (rainfed vs
map mix). Writes `irrigation_per_country.csv`.
"""

import os

import matplotlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = "/projects/prjs1788/tnp-crop"
TABLES = {"rainfed": f"{RUN}/wofost_eu_grid_dense_omfix/processed/wofost_maize_daily.parquet",
          "pp": f"{RUN}/wofost_eu_grid_dense_pp/processed/wofost_maize_daily.parquet",
          "mix": f"{RUN}/wofost_eu_grid_dense_irrmap/processed/wofost_maize_daily.parquet"}
CYBENCH = "/projects/prjs1788/CropBench/data/cybench/cybench-data/maize"
CODE = {"Austria": "AT", "Belgium": "BE", "Bulgaria": "BG", "Croatia": "HR", "Czechia": "CZ", "Denmark": "DK", "France": "FR",
        "Germany": "DE", "Greece": "EL", "Hungary": "HU", "Italy": "IT", "Lithuania": "LT", "Netherlands": "NL", "Poland": "PL",
        "Portugal": "PT", "Romania": "RO", "Slovakia": "SK", "Spain": "ES", "Sweden": "SE", "Slovenia": "SI"}
COLORS = {"irrigated": "#1b9e77", "rainfed": "#7a7a7a", "rainfed_no_map": "#d9d9d9"}


def cybench(code: str):
    path = f"{CYBENCH}/{code}/yield_maize_{code}.csv"
    if not os.path.exists(path):
        return None, None
    c = pd.read_csv(path)
    c = c[c["harvest_year"].between(2005, 2020) & (c["yield"] > 0)]
    if c.empty:
        return None, None
    if c["production"].notna().any():  # production / harvested area; where there is no production column, the mean regional yield
        national = c.groupby("harvest_year")["production"].sum() / c.groupby("harvest_year")["harvest_area"].sum()
    else:
        national = c.groupby("harvest_year")["yield"].mean()
    return c["yield"].mean(), national


def main():
    a = pd.read_csv(f"{HERE}/location_irrigation.csv")
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.1, 1]})
    for status, label in (("rainfed_no_map", "no maize area in the map cell (rainfed)"), ("rainfed", "rainfed (draw)"), ("irrigated", "irrigated (draw)")):
        d = a[a.status == status]
        ax[0].scatter(d.longitude, d.latitude, s=4, color=COLORS[status], label=f"{label}: {len(d)}", alpha=0.8, linewidths=0)
    ax[0].set_xlim(-10, 32)
    ax[0].set_ylim(35, 60)
    ax[0].set_title("MIRCA-OS 2015 maize map at the 12,382 locations", fontsize=10)
    ax[0].legend(fontsize=7, markerscale=3, frameon=False, loc="lower right")
    g = a.groupby("country").agg(n=("irrigated", "size"), irrigated_pct=("irrigated", lambda s: 100 * s.mean()),
                                 map_share_pct=("irrigated_share", lambda s: 100 * s.mean()),
                                 no_map_pct=("status", lambda s: 100 * (s == "rainfed_no_map").mean())).query("n >= 50").sort_values("irrigated_pct")
    y = np.arange(len(g))
    ax[1].barh(y - 0.2, g.irrigated_pct, 0.4, color=COLORS["irrigated"], label="locations drawn irrigated, %")
    ax[1].barh(y + 0.2, g.no_map_pct, 0.4, color=COLORS["rainfed_no_map"], label="locations without maize area in the map, %")
    ax[1].scatter(g.map_share_pct, y - 0.2, color="k", s=12, zorder=3, label="mean irrigated share of the in-map cells, %")
    ax[1].set_yticks(y, g.index, fontsize=8)
    ax[1].legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{HERE}/figures/irrigation_map.png", dpi=140)
    plt.close(fig)

    cols = ["location_index", "country", "sowing_year", "yield_t_per_ha", "reached_maturity"]
    sim = {k: pq.read_table(p, columns=cols).to_pandas() for k, p in TABLES.items()}
    for k, d in sim.items():
        sim[k] = d[d["reached_maturity"].astype(bool) & d["sowing_year"].between(2005, 2020)]
    rows, fig2 = [], None
    countries = [c for c in CODE if c in set(sim["mix"].country) and (sim["mix"].country == c).sum() > 3000]
    fig2, axes = plt.subplots(1, 2, figsize=(15, 5))
    x, w = np.arange(len(countries)), 0.2
    means = {k: [sim[k][sim[k].country == c].yield_t_per_ha.mean() for c in countries] for k in sim}
    cyb_mean, cyb_r = [], {"rainfed": [], "mix": []}
    for c in countries:
        m, nat = cybench(CODE[c])
        cyb_mean.append(m if m is not None else np.nan)
        for k in ("rainfed", "mix"):
            s = sim[k][sim[k].country == c].groupby("sowing_year").yield_t_per_ha.mean()
            common = s.index.intersection(nat.dropna().index) if nat is not None else []  # years where the reported series exists
            cyb_r[k].append(np.corrcoef(s.loc[common], nat.loc[common])[0, 1] if len(common) > 3 else np.nan)
    for i, (k, label, color) in enumerate((("rainfed", "rainfed (OM fix)", "#7a7a7a"), ("pp", "fully irrigated (PP)", "#9ad0c1"), ("mix", "map mix", "#1b9e77"))):
        axes[0].bar(x + (i - 1.5) * w, means[k], w, color=color, label=label)
    axes[0].bar(x + 1.5 * w, cyb_mean, w, color="#d95f02", label="CY-Bench reported")
    axes[0].set_xticks(x, [CODE[c] for c in countries])
    axes[0].set_ylabel("mean yield, t/ha")
    axes[0].legend(fontsize=8, frameon=False)
    axes[1].bar(x - 0.2, cyb_r["rainfed"], 0.4, color="#7a7a7a", label="rainfed")
    axes[1].bar(x + 0.2, cyb_r["mix"], 0.4, color="#1b9e77", label="map mix")
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].set_xticks(x, [CODE[c] for c in countries])
    axes[1].set_ylabel("r with reported national yield (year to year)")
    axes[1].legend(fontsize=8, frameon=False)
    fig2.suptitle("Simulated yield: rainfed vs fully irrigated vs MIRCA-OS map mix (2005-2020, matured seasons)")
    fig2.tight_layout()
    fig2.savefig(f"{HERE}/figures/irrigation_yields.png", dpi=140)
    pd.DataFrame({"country": countries, **{f"yield_{k}": means[k] for k in means}, "yield_cybench": cyb_mean,
                  "r_year_to_year_rainfed": cyb_r["rainfed"], "r_year_to_year_mix": cyb_r["mix"]}).round(3).to_csv(f"{HERE}/irrigation_per_country.csv", index=False)
    print(pd.read_csv(f"{HERE}/irrigation_per_country.csv").to_string())


if __name__ == "__main__":
    main()
