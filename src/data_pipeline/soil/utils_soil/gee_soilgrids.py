import logging
from typing import Optional

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.data_pipeline.soil.utils_soil.default_soil_variables import (
    default_gee_soil_config,
    default_soilgrid_d_factors,
    default_zs,
)

log = logging.getLogger(__name__)

# SoilGrids asset mapping
SOIL_ASSETS = {
    'clay': 'projects/soilgrids-isric/clay_mean',
    'nitrogen': 'projects/soilgrids-isric/nitrogen_mean',
    'phh2o': 'projects/soilgrids-isric/phh2o_mean',
    'soc': 'projects/soilgrids-isric/soc_mean',
    'bdod': 'projects/soilgrids-isric/bdod_mean',
    'sand': 'projects/soilgrids-isric/sand_mean',
    'silt': 'projects/soilgrids-isric/silt_mean',
}


def ensure_ee_initialized() -> None:
    """Initialize the Earth Engine client if it isn't already (idempotent).
    Assumes `ee.Authenticate()` has already been run once, outside this
    codebase, per the user's own GEE project setup.
    """
    import ee

    try:
        ee.Number(1).getInfo()
    except Exception:
        ee.Initialize(project='cropfm')


def soil(cfg: Optional[DictConfig], point, **kwargs) -> dict:
    """Extract SoilGrids soil properties for a point via Earth Engine.
    SoilGrids - 250m resolution.

    Args:
        cfg: config containing soil parameters (`log_level`, `name`,
            `variables`, `depth_layers`). Defaults to
            `default_soil_variables.default_gee_soil_config()` when omitted.
        point: Geographic point to extract data from (`ee.Geometry.Point`).
        **kwargs: Additional parameters (unused, kept for a uniform modality-
            extractor call signature).

    Returns:
        dict: `{"modality", "data", "variable_names"}`, where `data` is a
        DataFrame with one row per depth layer and one column per variable,
        already scaled to conventional units (see `default_soilgrid_d_factors`
        -- SoilGrids band values are raw mapped integers, e.g. clay in
        g/kg * 10; scaling happens here, upstream of everything else, so
        `calculate_van_genuchten` always receives conventional units and is
        never itself touched).
    """
    cfg = cfg if cfg is not None else OmegaConf.create(default_gee_soil_config())
    log.setLevel(cfg.log_level)
    variables = list(cfg.variables)
    depth_layers = list(cfg.depth_layers)

    log.info("Starting SoilGrids extraction")

    d_factors = default_soilgrid_d_factors()
    soil_data = {}

    for var in variables:
        log.debug(f"Processing soil property: {var}")

        soil_image = _import_ee().Image(SOIL_ASSETS[var])
        band_names = [f"{var}_{depth}_mean" for depth in depth_layers]

        pixel_value = soil_image.select(band_names).sample(region=point, scale=250, numPixels=1)
        sample_data = pixel_value.getInfo()

        if sample_data["features"]:
            factor = d_factors.get(var, 1)
            soil_data[var] = [
                sample_data["features"][0]["properties"][band_name] / factor for band_name in band_names
            ]

    df = pd.DataFrame(soil_data)
    soil_data_dict = {"modality": cfg.name, "data": df, "variable_names": variables}

    log.info(f"Successfully extracted {len(soil_data_dict['data'])} soil property observations")

    return soil_data_dict


def get_df_soilgrids_gee(cfg: Optional[DictConfig], longitude: float, latitude: float) -> pd.DataFrame:
    """GEE-based counterpart to `soilgrids.get_df_soilgrids()`: returns the
    same depth-indexed DataFrame shape (`latitude`, `longitude`, `zmin`,
    `zmax` + SoilGrids variables in conventional units) that
    `calculate_van_genuchten` expects, sourced from Earth Engine instead of
    the ISRIC REST API. `cfg=None` uses `default_gee_soil_config()`.
    Requires the `earthengine-api` package and an authenticated GEE project.
    """
    ensure_ee_initialized()
    point = _import_ee().Geometry.Point([longitude, latitude])
    soil_result = soil(cfg, point)

    df_raw = soil_result["data"]
    zmins, zmaxs = default_zs()
    if len(df_raw) != len(zmins):
        raise ValueError(
            f"Expected {len(zmins)} depth layers (matching default_zs()), got {len(df_raw)} "
            "from the GEE soil extraction -- check the depth_layers config."
        )

    df = pd.DataFrame({"latitude": latitude, "longitude": longitude, "zmin": zmins, "zmax": zmaxs})
    for var in df_raw.columns:
        df[var] = df_raw[var].to_numpy()
    return df


def _import_ee():
    # Local import so the rest of the data pipeline still works without
    # earthengine-api installed; only the GEE-based soil path needs it.
    import ee

    return ee
