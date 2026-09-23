import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import zarr

# Default list of North-Western European countries
DEFAULT_NW_EUROPE_COUNTRIES = [
    "Netherlands",
    "Germany",
    "France",
    "Belgium",
    "United Kingdom",
    "Ireland",
    "Luxembourg",
    "Denmark",
]


def load_sample_info_from_zarr(zarr_path: str) -> pd.DataFrame:
    """Load the metadata/sample_info structured array from a Zarr archive into a pandas DataFrame.

    :param zarr_path: Path to the root of the .zarr dataset directory or zip.
    :return: DataFrame containing sample metadata (sample_id, country, longitude, latitude, etc.).
    """
    path = Path(zarr_path)
    if not path.exists():
        raise FileNotFoundError(f"Zarr archive not found at: {zarr_path}")

    root = zarr.open(str(path), mode="r")

    if "metadata/sample_info" not in root:
        raise KeyError(
            f"'metadata/sample_info' array not found in Zarr at {zarr_path}. "
            f"Available keys under root: {list(root.keys())}"
        )

    # Read structured array into memory
    sample_info_arr = root["metadata/sample_info"][:]
    df = pd.DataFrame(sample_info_arr)

    # Decode byte strings if dtype is byte string (S-type) rather than unicode (U-type)
    for col in df.columns:
        if df[col].dtype == object or df[col].dtype.kind in ("S", "U"):
            if len(df) > 0 and isinstance(df[col].iloc[0], (bytes, bytearray)):
                df[col] = df[col].str.decode("utf-8")
            df[col] = df[col].astype(str).str.strip()

    return df


def filter_and_sample_locations(
    df: pd.DataFrame,
    countries: Optional[List[str]] = None,
    continent: Optional[str] = None,
    num_samples: Optional[int] = None,
    samples_per_country: Optional[int] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Filter sample locations by country/continent and optionally sample a subset.

    :param df: Input DataFrame loaded from sample_info.
    :param countries: List of country names to include. If None, no country filter is applied.
    :param continent: Optional continent name to filter by (e.g. 'Europe').
    :param num_samples: Total number of points to sample randomly across all matching countries.
        If None, all matching points are returned.
    :param samples_per_country: Number of points to sample per country.
        If specified, stratifies sampling evenly per country (up to available points).
    :param seed: Random seed for reproducibility.
    :return: Filtered and sampled DataFrame.
    """
    filtered_df = df.copy()

    # Filter by continent if specified
    if continent:
        filtered_df = filtered_df[
            filtered_df["continent"].str.lower() == continent.strip().lower()
        ]

    # Filter by country (case-insensitive)
    if countries:
        target_countries = [c.strip().lower() for c in countries]
        filtered_df = filtered_df[
            filtered_df["country"].str.lower().isin(target_countries)
        ]

    if filtered_df.empty:
        print("Warning: No samples matched the specified filter criteria.")
        return filtered_df

    # Sampling logic
    if samples_per_country is not None and samples_per_country > 0:
        # Stratified sampling: up to k samples per country
        sampled_dfs = []
        for country_name, group in filtered_df.groupby("country"):
            n_to_sample = min(len(group), samples_per_country)
            sampled_dfs.append(group.sample(n=n_to_sample, random_state=seed))
        filtered_df = pd.concat(sampled_dfs, ignore_index=True)

    elif num_samples is not None and num_samples > 0:
        # Global random sampling across all filtered points
        n_to_sample = min(len(filtered_df), num_samples)
        filtered_df = filtered_df.sample(n=n_to_sample, random_state=seed)

    # Sort deterministically by country and sample_id if available
    sort_cols = [c for c in ["country", "sample_id"] if c in filtered_df.columns]
    if sort_cols:
        filtered_df = filtered_df.sort_values(by=sort_cols).reset_index(drop=True)
    else:
        filtered_df = filtered_df.reset_index(drop=True)

    return filtered_df


def main():
    if len(sys.argv) == 1:
        print("No arguments provided!")
        print(
            "Usage: python extract_locations_from_zarr.py --zarr-path <path/to/raw_dataset.zarr> "
            "[--countries <country1> <country2> ... | --nw-europe] "
            "[--num-samples <n> | --samples-per-country <n>] "
            "[--seed <int>] [--output-csv <path>]"
        )
        print("\nExamples:")
        print(
            "  # 1. Extract ALL locations in NW Europe:\n"
            "  python extract_locations_from_zarr.py --zarr-path data/raw_dataset.zarr --nw-europe -o data/nw_europe_all.csv\n"
        )
        print(
            "  # 2. Randomly sample 500 points across Germany, France, and Netherlands:\n"
            "  python extract_locations_from_zarr.py --zarr-path data/raw_dataset.zarr --countries Germany France Netherlands -n 500 --seed 42 -o data/sampled_500.csv\n"
        )
        print(
            "  # 3. Sample 100 points per NW European country:\n"
            "  python extract_locations_from_zarr.py --zarr-path data/raw_dataset.zarr --nw-europe --samples-per-country 100 --seed 42 -o data/nw_europe_stratified.csv\n"
        )
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Extract and sample location coordinates and metadata from CropFM dataset Zarr."
    )
    parser.add_argument(
        "--zarr-path",
        dest="zarr_path",
        type=str,
        required=True,
        help="Path to raw_dataset.zarr directory.",
    )
    country_group = parser.add_mutually_exclusive_group()
    country_group.add_argument(
        "--countries",
        dest="countries",
        nargs="+",
        default=None,
        help="List of country names (space-separated, e.g. --countries Netherlands Germany France).",
    )
    country_group.add_argument(
        "--nw-europe",
        dest="nw_europe",
        action="store_true",
        help=f"Use default NW Europe countries: {', '.join(DEFAULT_NW_EUROPE_COUNTRIES)}.",
    )
    parser.add_argument(
        "--continent",
        dest="continent",
        type=str,
        default=None,
        help="Filter by continent name (e.g. Europe).",
    )

    sample_group = parser.add_mutually_exclusive_group()
    sample_group.add_argument(
        "-n",
        "--num-samples",
        dest="num_samples",
        type=int,
        default=100,
        help="Total number of points to sample randomly across all selected countries.",
    )
    sample_group.add_argument(
        "--samples-per-country",
        dest="samples_per_country",
        type=int,
        default=None,
        help="Number of points to sample per selected country (stratified).",
    )

    parser.add_argument(
        "--seed",
        dest="seed",
        type=int,
        default=None,
        help="Random seed for sampling reproducibility.",
    )
    parser.add_argument(
        "-o",
        "--output-csv",
        dest="output_csv",
        type=str,
        default="locations.csv",
        help="Output CSV path (default: locations.csv).",
    )

    args = parser.parse_args()

    # Determine countries
    selected_countries = None
    if args.nw_europe:
        selected_countries = DEFAULT_NW_EUROPE_COUNTRIES
    elif args.countries:
        selected_countries = args.countries

    print(f"Loading sample info from {args.zarr_path}...")
    df = load_sample_info_from_zarr(args.zarr_path)
    print(f"Loaded {len(df):,} total samples.")

    print("Filtering and sampling locations...")
    result_df = filter_and_sample_locations(
        df=df,
        countries=selected_countries,
        continent=args.continent,
        num_samples=args.num_samples,
        samples_per_country=args.samples_per_country,
        seed=args.seed,
    )

    if result_df.empty:
        print("No locations found matching criteria.")
        sys.exit(0)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result_df.to_csv(output_path, index=False)
    print(f"\nSaved {len(result_df):,} locations to: {output_path}")

    # Summary table
    if "country" in result_df.columns:
        print("\nBreakdown by country:")
        summary = result_df["country"].value_counts()
        for c_name, count in summary.items():
            print(f"  - {c_name}: {count:,}")


if __name__ == "__main__":
    main()
