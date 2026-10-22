"""Where the simulation locations are: the current `locations_maize.csv` vs the EU grid samples.

    uv run python -m analysis.plot_locations

Reads the `*_per_country.csv` files `grid_sample_locations.py` writes and the locations CSVs, and writes
`analysis/data/locations_per_country.csv` (points per country in the zarr, the current CSV, its simulated
first 1585 rows, and the two grid samples) and `analysis/figures/locations_eu_grid.png` (share of
locations per country, and maps).
"""

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LOC = "data/raw/locations"
SETS = {  # name -> (csv, label, color)
    "current": (f"{LOC}/locations_maize.csv", "current locations_maize.csv (NW Europe)", "#8a8984"),
    "grid_cap5": (f"{LOC}/locations_maize_eu_grid.csv", "EU grid, ≤5 cells per 1° block", "#2a78d6"),
    "grid_nocap": (f"{LOC}/locations_maize_eu_grid_nocap.csv", "EU grid, 2000 cells, no cap", "#eb6834"),
}
SIMULATED = 1585  # rows of locations_maize.csv with downloaded soil/weather


def main():
    frames = {name: pd.read_csv(path) for name, (path, _, _) in SETS.items()}
    steps = pd.read_csv(f"{LOC}/locations_maize_eu_grid_per_country.csv", index_col="country")
    counts = steps[["zarr_all_points", "zarr_maize_points", "eu_points", "cells_0.1deg"]].join(
        pd.DataFrame({"current_csv": frames["current"]["country"].value_counts(),
                      "current_simulated": frames["current"].iloc[:SIMULATED]["country"].value_counts(),
                      "grid_cap5": frames["grid_cap5"]["country"].value_counts(),
                      "grid_nocap": frames["grid_nocap"]["country"].value_counts()}), how="outer")
    counts = counts.fillna(0).astype(int).sort_values(["grid_cap5", "zarr_maize_points"], ascending=False)
    counts.to_csv("analysis/data/locations_per_country.csv")
    print(counts.to_string())

    shown = counts[(counts[["current_csv", "grid_cap5", "grid_nocap"]] > 0).any(axis=1)].index
    fig = plt.figure(figsize=(13, 10))
    for k, (name, (_, label, color)) in enumerate(SETS.items()):
        m = fig.add_axes([0.03 + k * 0.325, 0.43, 0.30, 0.52])
        d = frames[name]
        m.scatter(d["longitude"], d["latitude"], s=2, color=color, linewidths=0)
        if name == "current":
            sim = d.iloc[:SIMULATED]
            m.scatter(sim["longitude"], sim["latitude"], s=2, color="#1f1e1c", linewidths=0)
        m.set_xlim(-10, 30)
        m.set_ylim(35, 61)
        m.set_aspect(1.5)  # ~1 / cos(48 deg)
        m.set_xticks([])
        m.set_yticks([])
        m.set_title(f"{label}, n={len(d):,}" + ("\nblack = simulated so far (first 1,585 rows)" if name == "current" else ""),
                    loc="left", fontsize=9)

    ax = fig.add_axes([0.06, 0.12, 0.92, 0.27])
    width = 0.27
    for k, (name, (_, label, color)) in enumerate(SETS.items()):
        share = 100 * counts.loc[shown, "current_csv" if name == "current" else name] / len(frames[name])
        ax.bar([i + (k - 1) * width for i in range(len(shown))], share, width=width * 0.9, color=color,
               label=f"{label} (n={len(frames[name]):,})")
    ax.set_xticks(range(len(shown)), shown, rotation=45, ha="right")
    ax.set_xlim(-0.6, len(shown) - 0.4)
    ax.set_ylabel("share of locations (%)")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.set_title("locations per country (share of each set)", loc="left", fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.savefig("analysis/figures/locations_eu_grid.png", dpi=170)
    print("wrote analysis/figures/locations_eu_grid.png")


if __name__ == "__main__":
    main()
