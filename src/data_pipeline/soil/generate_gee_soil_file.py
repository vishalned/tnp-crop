import argparse
import os
import sys
from typing import Optional, Tuple

import rootutils
import yaml
from omegaconf import DictConfig

from src.data_pipeline.soil.utils_soil.gee_soilgrids import get_df_soilgrids_gee
from src.data_pipeline.soil.utils_soil.generate_soil_files import (
    calculate_van_genuchten,
    dump_soil_yaml,
    generate_df_soil_input,
    generate_soil_yaml,
)
from src.data_pipeline.soil.utils_soil.soil_static_features import compute_topsoil_static_features


root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
DEFAULT_SOIL_SAVE_DIR = os.path.join(str(root), "data", "raw", "soilgrids_gee")


def generate_soil_file_from_gee(
    longitude: float,
    latitude: float,
    output_dir: Optional[str] = None,
    soil_cfg: Optional[DictConfig] = None,
) -> dict:
    """GEE-based counterpart to
    `generate_soilgrids_soil_file.generate_soil_file`: the same van
    Genuchten -> pF-curve tabulation -> PCSE multi-layer soil YAML assembly
    (`generate_df_soil_input` / `generate_soil_yaml` / `dump_soil_yaml`,
    reused unchanged) applied to Earth Engine's full 6-depth SoilGrids pull
    instead of the ISRIC REST API. The REST-based script is left untouched
    in the repo as a reference/fallback, unused by the active pipeline.

    Returns a dict with the written YAML path and the topsoil-derived static
    features (`awc`, `bulk_density`) the multi-layer profile doesn't
    otherwise expose as single scalars for the whole soil column.

    `soil_cfg` defaults to `default_soil_variables.default_gee_soil_config()`
    when omitted (all 7 SoilGrids variables, the full 6-depth grid).
    """
    save_dir = output_dir if output_dir is not None else DEFAULT_SOIL_SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)

    df_soilgrids = get_df_soilgrids_gee(soil_cfg, longitude=longitude, latitude=latitude)

    vg_data = calculate_van_genuchten(df_soilgrids)
    df_soil_input = generate_df_soil_input(vg_data)
    soil_yaml = generate_soil_yaml(df_soil_input)
    static_features = compute_topsoil_static_features(df_soilgrids, vg_data)

    path_file = os.path.join(save_dir, f"soil_{longitude}_{latitude}.yaml")
    dump_soil_yaml(soil_yaml, path_file)

    print(f"YAML soil file has been created at {path_file}.")
    return {"path": path_file, "static_features": static_features}


def generate_soil_data_for_wofost(
    longitude: float,
    latitude: float,
    output_dir: Optional[str] = None,
    soil_cfg: Optional[DictConfig] = None,
) -> Tuple[dict, dict]:
    """`generate_soil_file_from_gee`, but returning the soil data ready to
    hand to PCSE's `ParameterProvider` instead of just a file path -- the
    equivalent of what `request_openmeteo_weather` already returns
    ready-to-use for weather, since a WOFOST run needs the parsed
    `SoilProfileDescription` dict, not the YAML file on disk.

    Returns `(soil_data, static_features)`: the parsed dict, and the
    topsoil-derived `awc`/`bulk_density` CYBench-aligned static features
    (the multi-layer profile doesn't otherwise expose single scalars for the
    whole soil column).
    """
    result = generate_soil_file_from_gee(longitude, latitude, output_dir, soil_cfg)
    with open(result["path"]) as f:
        soil_data = yaml.safe_load(f)
    return soil_data, result["static_features"]


def main():
    if len(sys.argv) == 1:
        print("No arguments provided!")
        print("Usage: python generate_gee_soil_file.py --lon <longitude> --lat <latitude> [--output-dir <path>]")
        print("Example: python generate_gee_soil_file.py -lon 6.656 -lat 52.966")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Generate a PCSE multi-layer soil YAML for a location via Earth Engine SoilGrids.")
    parser.add_argument("-lon", "--longitude", dest="longitude", type=float, required=True, help="Longitude for the soil data.")
    parser.add_argument("-lat", "--latitude", dest="latitude", type=float, required=True, help="Latitude for the soil data.")
    parser.add_argument("-o", "--output-dir", dest="output_dir", type=str, default=DEFAULT_SOIL_SAVE_DIR, help="Directory to save generated soil YAML files.")

    args = parser.parse_args()

    generate_soil_file_from_gee(
        longitude=args.longitude,
        latitude=args.latitude,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
