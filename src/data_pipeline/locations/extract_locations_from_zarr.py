"""Sample simulation locations from the CropFM dataset Zarr
(`raw_dataset.zarr` / `cleaned_dataset.zarr`, see CropFM's
`zarr_data_structure.md`) into a locations CSV for
`generate_wofost_dataset.py`.

Each Zarr sample is a point on WorldCereal temporary cropland. Besides its
`metadata/sample_info` (sample_id, country, continent, longitude, latitude,
...), the Zarr stores per sample:

- `static_modalities/worldcereal_cropmask/crop_mask`: the WorldCereal 2021
  crop type at the point -- 0 no crop, 1 maize, 2 winter cereals,
  3 spring cereals, -1 missing (where products overlap, maize takes
  precedence, then winter cereals; see CropFM's `worldcereal_cropmask.py`).
- `static_modalities/worldcereal_cropcalendar/crop_calendar`: the WorldCereal
  AEZ crop calendar, start/end of season (day of year) per crop type.

`--crop` keeps only points where WorldCereal maps that crop (wheat = winter
cereals, matching the winter-wheat variety the WOFOST runner uses) and adds
a `crop` column, which `generate_wofost_dataset.py` reads per row, plus that
crop's `sos_doy`/`eos_doy` from the crop calendar.
"""

import argparse
import os
import sys
from typing import List, Optional

import numpy as np
import pandas as pd
import rootutils


root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
DEFAULT_LOCATIONS_DIR = os.path.join(str(root), "data", "raw", "locations")


def default_country_groups() -> dict:
    """Named country lists usable in `--countries` instead of country names."""
    return {
        "nw_europe": [
            "Netherlands",
            "Germany",
            "France",
            "Belgium",
            "United Kingdom",
            "Ireland",
            "Luxembourg",
            "Denmark",
        ],
    }


def worldcereal_crop_codes() -> dict:
    """WOFOST crop name -> WorldCereal `crop_mask` code it is sampled from.
    Wheat maps to winter cereals because the runner simulates winter wheat
    (`Winter_wheat_101`, sown in autumn)."""
    return {"maize": 1, "wheat": 2}


def worldcereal_calendar_season() -> dict:
    """WOFOST crop name -> WorldCereal crop calendar season prefix."""
    return {"maize": "tc-maize-main", "wheat": "tc-wintercereals"}


def _decode_strings(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype.kind in ("S", "O", "U"):
            if len(df) and isinstance(df[col].iloc[0], (bytes, bytearray)):
                df[col] = df[col].str.decode("utf-8")
            df[col] = df[col].astype(str).str.strip()
    return df


def load_locations_from_zarr(zarr_path: str) -> pd.DataFrame:
    """All samples in the Zarr as a DataFrame: `metadata/sample_info` plus
    `zarr_index` (row in the Zarr), `worldcereal_crop_mask` and the crop
    calendar columns (e.g. `tc-maize-main_sos`, day of year, NaN if
    missing) when the Zarr has them."""
    import zarr

    if not os.path.exists(zarr_path):
        raise FileNotFoundError(f"Zarr archive not found at: {zarr_path}")
    store = zarr.open(zarr_path, mode="r")

    df = _decode_strings(pd.DataFrame(store["metadata/sample_info"][:]))
    df.insert(0, "zarr_index", np.arange(len(df)))

    try:
        df["worldcereal_crop_mask"] = store["static_modalities/worldcereal_cropmask/crop_mask"][:]
    except KeyError:
        print("Warning: no WorldCereal crop mask in this Zarr; --crop filtering is unavailable.")

    try:
        calendar = store["static_modalities/worldcereal_cropcalendar/crop_calendar"][:].astype(float)
        names = [str(n) for n in store["static_modalities/worldcereal_cropcalendar/variable_names"][:]]
        calendar[calendar < 0] = np.nan  # -1 = missing
        for i, name in enumerate(names):
            df[name] = calendar[:, i]
    except KeyError:
        pass

    return df


def _resolve_countries(countries: Optional[List[str]]) -> Optional[List[str]]:
    if not countries:
        return None
    groups = default_country_groups()
    resolved = []
    for c in countries:
        resolved += groups.get(c.lower(), [c])
    return resolved


def sample_locations(
    df: pd.DataFrame,
    crop: Optional[str] = None,
    countries: Optional[List[str]] = None,
    num_locations: int = 100,
    per_country: bool = False,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Filter by crop and countries, then randomly sample.

    :param crop: `wheat` or `maize`: keep only points WorldCereal maps as
        that crop, and add `crop`, `sos_doy`, `eos_doy` columns.
    :param countries: country names (case-insensitive) or group names from
        `default_country_groups()`, e.g. `["nw_europe"]`. None = all.
    :param num_locations: how many to sample (in total, or per country with
        `per_country=True`). 0 keeps every matching point.
    """
    selected = df
    if crop is not None:
        if "worldcereal_crop_mask" not in selected:
            raise ValueError("This Zarr has no WorldCereal crop mask, so --crop can't be applied.")
        selected = selected[selected["worldcereal_crop_mask"] == worldcereal_crop_codes()[crop]].copy()
        selected.insert(selected.columns.get_loc("latitude") + 1, "crop", crop)
        season = worldcereal_calendar_season()[crop]
        for bound in ("sos", "eos"):
            if f"{season}_{bound}" in selected:
                selected[f"{bound}_doy"] = selected[f"{season}_{bound}"]

    countries = _resolve_countries(countries)
    if countries:
        wanted = {c.lower() for c in countries}
        unknown = wanted - set(df["country"].str.lower())
        if unknown:
            print(f"Warning: no samples at all for countries {sorted(unknown)} (check spelling).")
        selected = selected[selected["country"].str.lower().isin(wanted)]

    if num_locations > 0:
        if per_country:
            selected = pd.concat(
                [g.sample(n=min(len(g), num_locations), random_state=seed) for _, g in selected.groupby("country")]
            )
        else:
            selected = selected.sample(n=min(len(selected), num_locations), random_state=seed)

    return selected.sort_values(["country", "zarr_index"]).reset_index(drop=True)


def main():
    if len(sys.argv) == 1:
        print("No arguments provided!")
        print(
            "Usage: python extract_locations_from_zarr.py --zarr-path <path/to/dataset.zarr> "
            "[--crop <wheat|maize>] [--countries <name|nw_europe> ...] "
            "[-n <num locations, 0 = all>] [--per-country] [--seed <int>] [-o <output csv>]"
        )
        print(
            "Example: python extract_locations_from_zarr.py --zarr-path data/raw_dataset.zarr "
            "--crop maize --countries nw_europe -n 500 --seed 42"
        )
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Sample simulation locations from the CropFM dataset Zarr.")
    parser.add_argument("--zarr-path", dest="zarr_path", type=str, default="/projects/prjs1788/CropFM/europe_data_1200_merged_cleaned2_rechunked.zarr", help="Path to the CropFM dataset .zarr.")
    parser.add_argument("--crop", dest="crop", type=str, default=None, choices=sorted(worldcereal_crop_codes()), help="Only points WorldCereal maps as this crop (wheat = winter cereals).")
    parser.add_argument("--countries", dest="countries", nargs="+", default=None, help=f"Country names or groups ({', '.join(default_country_groups())}). Default: all.")
    parser.add_argument("-n", "--num-locations", dest="num_locations", type=int, default=100, help="Number of locations to sample (per country with --per-country); 0 keeps all. Default 100.")
    parser.add_argument("--per-country", dest="per_country", action="store_true", help="Sample -n locations from each country instead of -n in total.")
    parser.add_argument("--seed", dest="seed", type=int, default=None)
    parser.add_argument("-o", "--output-csv", dest="output_csv", type=str, default=None, help="Default: data/raw/locations/locations_{crop or all}.csv")

    args = parser.parse_args()

    print(f"Loading sample info from {args.zarr_path}...")
    df = load_locations_from_zarr(args.zarr_path)
    print(f"Loaded {len(df):,} samples.")

    locations = sample_locations(
        df,
        crop=args.crop,
        countries=args.countries,
        num_locations=args.num_locations,
        per_country=args.per_country,
        seed=args.seed,
    )
    if locations.empty:
        print("No locations matched.")
        sys.exit(1)

    output_csv = args.output_csv or os.path.join(DEFAULT_LOCATIONS_DIR, f"locations_{args.crop or 'all'}.csv")
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    locations.to_csv(output_csv, index=False)

    print(f"Saved {len(locations):,} locations to {output_csv}.")
    for country, count in locations["country"].value_counts().items():
        print(f"  {country}: {count:,}")


if __name__ == "__main__":
    main()
