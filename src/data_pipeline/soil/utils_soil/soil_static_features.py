import pandas as pd

from src.data_pipeline.soil.utils_soil.default_soil_variables import (
    default_pf_field_capacity,
    default_pf_wilting_point,
)
from src.data_pipeline.soil.utils_soil.generate_soil_files import calculate_soil_moisture_content


def compute_topsoil_static_features(df_soilgrids: pd.DataFrame, vg_data: pd.DataFrame) -> dict:
    """CYBench-aligned static soil features (`awc`, `bulk_density`) read off
    the topsoil (shallowest) layer.

    The multi-layer waterbalance doesn't expose a single SMFCF/SMW/bulk
    density scalar for the whole profile the way the single-bucket classic
    waterbalance did -- per the MLWB soil-step spec, these are derived
    per-layer, and here specifically from the topsoil layer.
    """
    top_raw = df_soilgrids.iloc[0]
    top_vg = vg_data.iloc[0]

    # bulk_density: SoilGrids' own bdod value (`bdod_mean` asset), read
    # straight off the raw fetched profile -- not computed.
    bulk_density = float(top_raw.bdod)

    # awc: SoilGrids has no direct "available water capacity" field, so this
    # one is computed manually -- SMFCF - SMW from the van Genuchten
    # water-retention curve at field capacity / wilting point pF.
    smfcf = calculate_soil_moisture_content(
        default_pf_field_capacity(), top_vg.alpha, top_vg.n, top_vg.theta_r, top_vg.theta_s
    )
    smw = calculate_soil_moisture_content(
        default_pf_wilting_point(), top_vg.alpha, top_vg.n, top_vg.theta_r, top_vg.theta_s
    )
    awc = float(smfcf - smw)

    return {"awc": awc, "bulk_density": bulk_density}
