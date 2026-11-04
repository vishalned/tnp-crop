"""Irrigated or rainfed per location, from the MIRCA-OS maize map (2015, 5 arcmin).

    uv run --with tifffile python -m analysis.irrigation_map.assign_irrigation

MIRCA-OS (Kebede et al., Sci Data 2025; HydroShare 60a890eb841c460192c03bb590687145) gives the annual harvested area (ha) of
irrigated and rainfed maize per 5' cell for 2000, 2005, 2010 and 2015. The newest layer (2015) is used for every season.
For each location: the cell's irrigated and rainfed maize area, the irrigated share f = ir / (ir + rf), and its status:

- in the map: the cell has maize area (ir + rf > 0). The location is irrigated with probability f (one seeded Bernoulli draw
  per location, the same in every year); the draw is kept so the assignment is reproducible;
- not in the map (no maize area in its cell): kept rainfed (status "rainfed_no_map").

Writes `location_irrigation.csv` (row = location_index of the locations CSV) and prints the counts.
"""

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MIRCA = "/projects/prjs1788/tnp-crop/data/raw/mirca/maize_2015"
LOCATIONS = "data/raw/locations/locations_maize_eu27_dense700_s42.csv"
CELL = 1 / 12  # 5 arcmin
NODATA = -9


def read_grid(system: str) -> np.ndarray:
    import tifffile

    x = tifffile.imread(f"{MIRCA}/MIRCA-OS_Maize_2015_{system}.tif").astype(np.float64)
    x[x == NODATA] = np.nan
    return x


def cell_values(grid: np.ndarray, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    col = np.clip(np.floor((lon + 180) / CELL).astype(int), 0, grid.shape[1] - 1)
    row = np.clip(np.floor((90 - lat) / CELL).astype(int), 0, grid.shape[0] - 1)
    return grid[row, col]


def assign(locations: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    ir = np.nan_to_num(cell_values(read_grid("ir"), locations["longitude"].to_numpy(), locations["latitude"].to_numpy()))
    rf = np.nan_to_num(cell_values(read_grid("rf"), locations["longitude"].to_numpy(), locations["latitude"].to_numpy()))
    total = ir + rf
    f = np.where(total > 0, ir / np.where(total > 0, total, 1), np.nan)
    draw = np.random.default_rng(seed).random(len(locations))
    irrigated = (total > 0) & (draw < np.nan_to_num(f))
    status = np.where(total <= 0, "rainfed_no_map", np.where(irrigated, "irrigated", "rainfed"))
    return pd.DataFrame({"location_index": np.arange(len(locations)), "country": locations["country"].to_numpy(),
                         "longitude": locations["longitude"].to_numpy(), "latitude": locations["latitude"].to_numpy(),
                         "mirca_irrigated_ha": ir, "mirca_rainfed_ha": rf, "irrigated_share": f, "draw": draw,
                         "irrigated": irrigated, "status": status})


def main():
    loc = pd.read_csv(LOCATIONS)
    d = assign(loc)
    d.to_csv(f"{HERE}/location_irrigation.csv", index=False)
    n = len(d)
    in_map = d["status"] != "rainfed_no_map"
    print(f"{n} locations; in the map (maize area in the 5' cell): {int(in_map.sum())} ({100 * in_map.mean():.1f} %); no maize area in the cell: {int((~in_map).sum())}")
    print(f"status: {d['status'].value_counts().to_dict()}")
    g = d.groupby("country").agg(locations=("irrigated", "size"), in_map=("status", lambda s: int((s != "rainfed_no_map").sum())),
                                 mean_irrigated_share=("irrigated_share", "mean"), irrigated=("irrigated", "sum"))
    g["irrigated_pct"] = 100 * g["irrigated"] / g["locations"]
    print(g.round(2).to_string())


if __name__ == "__main__":
    main()
