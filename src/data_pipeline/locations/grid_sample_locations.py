"""Spatially even simulation locations: one point per 0.1 deg maize cell (one ERA5-Land pixel), over the EU.

    uv run python -m src.data_pipeline.locations.grid_sample_locations --crop maize -n 2000 --seed 42 \
        [--holdout-countries ...] [-o data/raw/locations/locations_maize_eu27_cell01_n2000_s42.csv]

Steps (only the grid is used, nothing from any benchmark):

1. Points WorldCereal maps as the crop, in the `--countries` (default `eu27`), minus `--holdout-countries`
   and points without a crop calendar.
2. Each point goes to its 0.1 deg cell (`cell_id`). Cells are centred on multiples of 0.1 deg, like the
   ERA5-Land grid, so one cell = one weather pixel. One random point per cell is kept, so every cell where
   the crop grows counts once, whatever the number of reference points in it.
3. If more cells are left than `-n`, `-n` cells are sampled uniformly at random. Locations then follow
   where the crop is grown, spread evenly in space.

All random choices use `--seed`. The output has the same columns as `extract_locations_from_zarr.py`'s
CSV (so `generate_wofost_dataset.py` reads it unchanged) plus `cell_id`. A sibling
`<output>_per_country.csv` counts the points per country at every step.
"""

import argparse
import os

import numpy as np
import pandas as pd

from src.data_pipeline.locations.extract_locations_from_zarr import (
    DEFAULT_LOCATIONS_DIR,
    load_locations_from_zarr,
    sample_locations,
)

ZARR_PATH = "/projects/prjs1788/CropFM/europe_data_1200_merged_cleaned2_rechunked.zarr"


def cell_ids(lat: pd.Series, lon: pd.Series, size_deg: float) -> pd.Series:
    """Id `"<lat index>_<lon index>"` of the `size_deg` cell (centred on multiples of `size_deg`) of each point."""
    i = np.round(lat.to_numpy() / size_deg).astype(int)
    j = np.round(lon.to_numpy() / size_deg).astype(int)
    return pd.Series([f"{a}_{b}" for a, b in zip(i, j)], index=lat.index)


def grid_sample(points: pd.DataFrame, num_cells: int, seed: int, cell_deg: float = 0.1) -> dict:
    """Steps 2-3 of the module docstring. Returns the points after each step."""
    rng = np.random.default_rng(seed)
    points = points.assign(cell_id=cell_ids(points["latitude"], points["longitude"], cell_deg))
    cells = points.iloc[rng.permutation(len(points))].drop_duplicates("cell_id")  # one random point per cell
    selected = cells
    if num_cells and len(cells) > num_cells:
        selected = cells.iloc[np.sort(rng.choice(len(cells), num_cells, replace=False))]
    order = ["country", "zarr_index"]
    return {name: df.sort_values(order).reset_index(drop=True)
            for name, df in (("points", points), ("cells", cells), ("selected", selected))}


def main():
    p = argparse.ArgumentParser(description="One point per 0.1 deg crop cell, sampled evenly over the EU.")
    p.add_argument("--zarr-path", default=ZARR_PATH)
    p.add_argument("--crop", default="maize")
    p.add_argument("--countries", nargs="+", default=["eu27"], help="Country names or groups (default eu27).")
    p.add_argument("--holdout-countries", nargs="+", default=[], help="Countries left out entirely.")
    p.add_argument("-n", "--num-cells", type=int, default=2000, help="Cell budget; 0 keeps every cell.")
    p.add_argument("--cell-deg", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("-o", "--output-csv", default=None,
                   help="Default: data/raw/locations/locations_{crop}_eu27_cell01_n{n}_s{seed}.csv")
    args = p.parse_args()

    zarr = load_locations_from_zarr(args.zarr_path)
    points = sample_locations(zarr, crop=args.crop, countries=args.countries, num_locations=0)
    holdout = {c.lower() for c in args.holdout_countries}
    points = points[~points["country"].str.lower().isin(holdout)]
    no_calendar = points["sos_doy"].isna() | points["eos_doy"].isna()
    print(f"{len(points):,} {args.crop} points; dropping {int(no_calendar.sum()):,} without a crop calendar.")
    points = points[~no_calendar]

    steps = grid_sample(points, args.num_cells, args.seed, args.cell_deg)
    output = args.output_csv or os.path.join(DEFAULT_LOCATIONS_DIR, f"locations_{args.crop}_eu27_cell01_n{args.num_cells}_s{args.seed}.csv")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    steps["selected"].to_csv(output, index=False)

    counts = pd.DataFrame({
        "zarr_all_points": zarr["country"].value_counts(),
        f"zarr_{args.crop}_points": zarr[zarr["worldcereal_crop_mask"] == points["worldcereal_crop_mask"].iloc[0]]
        ["country"].value_counts(),
        **{f"{name}": df["country"].value_counts() for name, df in steps.items()},
    }).fillna(0).astype(int).rename(columns={"points": "eu_points", "cells": "cells_0.1deg"})
    counts = counts.sort_values(["selected", f"zarr_{args.crop}_points"], ascending=False).rename_axis("country")
    counts.to_csv(output.replace(".csv", "_per_country.csv"))
    print(counts.to_string())
    print(f"Saved {len(steps['selected']):,} locations to {output}.")


if __name__ == "__main__":
    main()
