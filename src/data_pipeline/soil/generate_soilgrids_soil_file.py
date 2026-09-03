import argparse
import os
import sys
from pathlib import Path
import rootutils

from src.data_pipeline.soil.utils_soil.soilgrids import get_df_soilgrids
from src.data_pipeline.soil.utils_soil.generate_soil_files import (
    calculate_van_genuchten,
    generate_df_soil_input,
    generate_soil_yaml,
    dump_soil_yaml,
)


root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
DEFAULT_SOIL_SAVE_DIR = os.path.join(str(root), "data", "raw", "soilgrids")


def generate_soil_file(longitude: float, latitude: float, output_dir: str = None) -> str:
    """
    Generate a YAML soil file for given longitude and latitude.

    Args:
        longitude (float): Longitude value.
        latitude (float): Latitude value.
        output_dir (str, optional): Target directory to save the YAML file.
                                   Defaults to DEFAULT_SOIL_SAVE_DIR.

    Returns:
        str: Absolute path to the generated YAML file.
    """
    save_dir = output_dir if output_dir is not None else DEFAULT_SOIL_SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)

    # Get soil data from SoilGrids based on longitude and latitude
    soil_data = get_df_soilgrids(lat=latitude, lon=longitude)

    vg_data = calculate_van_genuchten(soil_data)

    df_soil_input = generate_df_soil_input(vg_data)

    soil_yaml = generate_soil_yaml(df_soil_input)

    # Write the soil data to a YAML file
    path_file = os.path.join(save_dir, f"soil_{longitude}_{latitude}.yaml")
    dump_soil_yaml(soil_yaml, path_file)

    print(f"YAML soil file has been created at {path_file}.")
    return path_file


def main():
    if len(sys.argv) == 1:
        print("No arguments provided!")
        print("Usage: python generate_soilgrids_soil_file.py --lon <longitude> --lat <latitude> [--output-dir <path>]")
        print("Example: python generate_soilgrids_soil_file.py -lon 6.656 -lat 52.966")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Generate a YAML soil file for a given longitude and latitude.")
    parser.add_argument("-lon", "--longitude", dest="longitude", type=float, required=True, help="Longitude for the soil data.")
    parser.add_argument("-lat", "--latitude", dest="latitude", type=float, required=True, help="Latitude for the soil data.")
    parser.add_argument("-o", "--output-dir", dest="output_dir", type=str, default=DEFAULT_SOIL_SAVE_DIR, help="Directory to save generated soil YAML files.")

    args = parser.parse_args()

    # Generate the YAML file
    generate_soil_file(
        longitude=args.longitude,
        latitude=args.latitude,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
