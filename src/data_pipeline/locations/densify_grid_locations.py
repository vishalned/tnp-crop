"""One-off: add points to the EU grid locations so every country has ~`--target-usable` usable points.

    uv run python -m src.data_pipeline.locations.densify_grid_locations \
        --existing-csv data/raw/locations/locations_maize_eu27_cell01_n2000_s42.csv \
        --table /projects/prjs1788/tnp-crop/wofost_eu_grid/processed/wofost_maize_daily.parquet

Why: a training batch is 8 tables of the same shape, and only countries with at least the batch's
`n_points` can fill it, so small countries get few tables (`docs/country_share_report.md`). With >= 682
usable points (4096 rows / 6 rows per point) every country can fill every shape.

Steps, per country of the existing EU grid run:

1. usable fraction u = existing points that matured in all 12 train years / existing points sampled
   (from the processed table, so whole-location failures count as unusable);
2. points wanted N = ceil(target / u), capped at the country's maize points; countries with u = 0
   (nothing usable, e.g. Latvia, Sweden) get nothing new;
3. points per ERA5-Land pixel k = max(1, ceil(N / pixels)), pixels = the country's 0.1 deg ERA5-Land
   pixels with maize points (`era5_land_cell`: edges at multiples of 0.1 deg);
4. pixels in a seeded random order, those that already have an existing point first; each pixel is
   filled up to k points (the existing point counts) until the country has N. Pixels with fewer than k
   points leave the country short, so then further passes allow k + 1, k + 2, ... per pixel.

New points never repeat an existing one, and avoid points within `CELL_EDGE_MARGIN_DEG` of a pixel
edge (those can't reuse the pixel's weather). The output is a copy of `--existing-csv` with the new
rows appended (the existing rows keep their row number = `location_index`), plus `era5_pixel`; a
sibling `_per_country.csv` lists every step's numbers.
"""

import argparse
import math

import numpy as np
import pandas as pd

from src.data_pipeline.locations.extract_locations_from_zarr import load_locations_from_zarr, sample_locations
from src.data_pipeline.locations.grid_sample_locations import ZARR_PATH, cell_ids
from src.data_pipeline.weather.utils_weather.gee_weather import era5_land_cell

TRAIN_YEARS = (2005, 2016)


def usable_fraction(existing: pd.DataFrame, table_path: str) -> pd.Series:
    """Per country: share of the existing points with a matured season in every train year."""
    df = pd.read_parquet(table_path, columns=["location_index", "sowing_year", "yield_t_per_ha", "reached_maturity"])
    train = df[df["sowing_year"].between(*TRAIN_YEARS) & df["reached_maturity"].astype(bool)
               & np.isfinite(df["yield_t_per_ha"])]
    years = train.groupby("location_index")["sowing_year"].nunique()
    full = years.index[years == TRAIN_YEARS[1] - TRAIN_YEARS[0] + 1]
    usable = existing.loc[full, "country"].value_counts()
    return (usable.reindex(existing["country"].unique()).fillna(0) / existing["country"].value_counts()).rename("usable")


def densify(points: pd.DataFrame, existing: pd.DataFrame, usable: pd.Series, target: int, seed: int):
    rng = np.random.default_rng(seed)
    points = points.assign(era5_pixel=[era5_land_cell(lon, lat) for lon, lat in zip(points["longitude"], points["latitude"])])
    existing = existing.assign(era5_pixel=[era5_land_cell(lon, lat) for lon, lat in zip(existing["longitude"], existing["latitude"])])
    candidates = points[~points["zarr_index"].isin(existing["zarr_index"]) & points["era5_pixel"].notna()]
    new, stats = [], []
    for country in existing["country"].value_counts().index:
        have = existing[existing["country"] == country]
        pool = candidates[candidates["country"] == country]
        available = len(have) + len(pool)
        u = float(usable.get(country, 0.0))
        wanted = min(math.ceil(target / u), available) if u > 0 else len(have)
        pixels = pd.Index(pool["era5_pixel"]).union(pd.Index(have["era5_pixel"].dropna())).unique()
        k = max(1, math.ceil(wanted / len(pixels)))
        old_pixels = have["era5_pixel"].dropna().unique()
        order = list(rng.permutation(old_pixels)) + list(rng.permutation(pixels.difference(old_pixels)))
        per_pixel = have["era5_pixel"].value_counts().to_dict()
        left = {p: g.iloc[rng.permutation(len(g))] for p, g in pool.groupby("era5_pixel")}  # random order per pixel
        total, added, cap = len(have), [], k
        while total < wanted and any(len(g) for g in left.values()):  # pixels with fewer points: next pass, cap + 1
            for pixel in order:
                g = left.get(pixel)
                take = min(cap - per_pixel.get(pixel, 0), wanted - total, 0 if g is None else len(g))
                if take > 0:
                    added.append(g.iloc[:take])
                    left[pixel], per_pixel[pixel], total = g.iloc[take:], per_pixel.get(pixel, 0) + take, total + take
            cap += 1
        added = pd.concat(added) if added else pool.iloc[:0]
        new.append(added)
        stats.append({"country": country, "maize_points": available, "era5_pixels": len(pixels),
                      "usable_fraction": round(u, 3), "wanted": wanted, "points_per_pixel": k, "max_per_pixel": max(per_pixel.values(), default=0),
                      "existing": len(have), "new": len(added), "total": len(have) + len(added),
                      "pixels_used": pd.concat([have["era5_pixel"], added["era5_pixel"]]).nunique(),
                      "expected_usable": int(round((len(have) + len(added)) * u))})
    return pd.concat(new), pd.DataFrame(stats)


def main():
    p = argparse.ArgumentParser(description="Add points per ERA5-Land pixel so every country has enough usable points.")
    p.add_argument("--existing-csv", default="data/raw/locations/locations_maize_eu27_cell01_n2000_s42.csv")
    p.add_argument("--table", default="/projects/prjs1788/tnp-crop/wofost_eu_grid/processed/wofost_maize_daily.parquet",
                   help="Processed table of the existing locations (for the usable fraction).")
    p.add_argument("--zarr-path", default=ZARR_PATH)
    p.add_argument("--crop", default="maize")
    p.add_argument("--target-usable", type=int, default=700)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("-o", "--output-csv", default="data/raw/locations/locations_maize_eu27_dense700_s42.csv")
    args = p.parse_args()

    # round_trip: the default parser can be one ulp off (the text of the existing rows is copied unchanged below)
    existing = pd.read_csv(args.existing_csv, float_precision="round_trip")
    usable = usable_fraction(existing, args.table)
    points = sample_locations(load_locations_from_zarr(args.zarr_path), crop=args.crop, countries=["eu27"], num_locations=0)
    points = points[points["sos_doy"].notna() & points["eos_doy"].notna()]

    new, stats = densify(points, existing, usable, args.target_usable, args.seed)
    new = new.assign(cell_id=cell_ids(new["latitude"], new["longitude"], 0.1)).sort_values(["country", "zarr_index"])
    # the existing rows are copied as text, so the batch runner parses them to the same coordinates as
    # before (its caches are named after them); only `era5_pixel` is added
    text = pd.read_csv(args.existing_csv, dtype=str, keep_default_na=False)
    text["era5_pixel"] = [era5_land_cell(lon, lat) or "" for lon, lat in zip(existing["longitude"], existing["latitude"])]
    assert not new["zarr_index"].isin(existing["zarr_index"]).any()
    text.to_csv(args.output_csv, index=False)
    new[text.columns].to_csv(args.output_csv, index=False, header=False, mode="a")
    stats.to_csv(args.output_csv.replace(".csv", "_per_country.csv"), index=False)
    print(stats.to_string(index=False))
    print(f"{len(existing):,} existing + {len(new):,} new = {len(existing) + len(new):,} locations -> {args.output_csv}")


if __name__ == "__main__":
    main()
