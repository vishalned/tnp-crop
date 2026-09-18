import argparse
import os
import sys
from typing import Optional

import rootutils
from omegaconf import DictConfig, OmegaConf

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
DEFAULT_GEE_SOIL_CONFIG = os.path.join(str(root), "configs", "data_pipeline", "soil", "gee_soil.yaml")


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
    """
    save_dir = output_dir if output_dir is not None else DEFAULT_SOIL_SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)

    cfg = soil_cfg if soil_cfg is not None else OmegaConf.load(DEFAULT_GEE_SOIL_CONFIG)
    df_soilgrids = get_df_soilgrids_gee(cfg, longitude=longitude, latitude=latitude)

    vg_data = calculate_van_genuchten(df_soilgrids)
    df_soil_input = generate_df_soil_input(vg_data)
    soil_yaml = generate_soil_yaml(df_soil_input)
    static_features = compute_topsoil_static_features(vg_data)

    path_file = os.path.join(save_dir, f"soil_{longitude}_{latitude}.yaml")
    dump_soil_yaml(soil_yaml, path_file)

    print(f"YAML soil file has been created at {path_file}.")
    return {"path": path_file, "static_features": static_features}


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
