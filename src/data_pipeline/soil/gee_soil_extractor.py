import ee
import pandas as pd
import logging
from omegaconf import DictConfig

from src.data_pipeline.soil.utils_soil.default_soil_variables import default_soilgrid_d_factors

log = logging.getLogger(__name__)


def soil(
    cfg: DictConfig,
    point: ee.Geometry.Point,
    **kwargs
) -> pd.DataFrame:
    """
    Extract SoilGrids soil properties data for a point.
    SoilGrids - 250m resolution

    Args:
        cfg: Hydra config containing soil parameters
        point: Geographic point to extract data from
        **kwargs: Additional parameters (can override config)

    Returns:
        pd.DataFrame: Extracted soil properties data with metadata
    """

    # Get config values with optional overrides from kwargs
    log.setLevel(cfg.log_level)
    variables = list(cfg.variables)
    depth_layers = list(cfg.depth_layers)

    log.info(f"Starting SoilGrids extraction")

    # SoilGrids asset mapping
    soil_assets = {
        'clay': 'projects/soilgrids-isric/clay_mean',
        'nitrogen': 'projects/soilgrids-isric/nitrogen_mean',
        'phh2o': 'projects/soilgrids-isric/phh2o_mean',
        'soc': 'projects/soilgrids-isric/soc_mean',
        'bdod': 'projects/soilgrids-isric/bdod_mean',
        'sand': 'projects/soilgrids-isric/sand_mean',
        'silt': 'projects/soilgrids-isric/silt_mean',
    }

    # SoilGrids band values are raw mapped integers (e.g. clay in g/kg * 10);
    # scale them to conventional units right here, upstream of everything
    # else (calculate_van_genuchten expects conventional units and is left
    # untouched -- see default_soilgrid_d_factors for the fixed, ISRIC-
    # documented conversion factors this applies).
    d_factors = default_soilgrid_d_factors()

    soil_data = {}

    for var in variables:
        log.debug(f"Processing soil property: {var}")

        # Load the soil property image
        soil_image = ee.Image(soil_assets[var])

        band_names = [f"{var}_{depth}_mean" for depth in depth_layers]

        pixel_value = soil_image.select(band_names).sample(
            region=point,
            scale=250,
            numPixels=1
        )

        sample_data = pixel_value.getInfo()

        if sample_data['features']:
            factor = d_factors.get(var, 1)
            soil_data[var] = [
                sample_data['features'][0]['properties'][band_name] / factor for band_name in band_names
            ]

    df = pd.DataFrame(soil_data)
    soil_data_dict = {
        'modality': cfg.name,
        'data': df,
        'variable_names': variables,
    }

    log.info(f"Successfully extracted {len(soil_data_dict['data'])} soil property observations")

    return soil_data_dict
