import pandas as pd

from src.data_pipeline.soil.utils_soil.default_soil_variables import (
    default_pf_field_capacity,
    default_pf_wilting_point,
)
from src.data_pipeline.soil.utils_soil.generate_soil_files import calculate_soil_moisture_content


def compute_topsoil_static_features(vg_data: pd.DataFrame) -> dict:
    """CYBench-aligned static soil features (`awc`, `bulk_density`) read off
    the topsoil (shallowest) van Genuchten layer.

    The multi-layer waterbalance doesn't expose a single SMFCF/SMW/bulk
    density scalar for the whole profile the way the single-bucket classic
    waterbalance did -- per the MLWB soil-step spec, these are derived
    per-layer, and here specifically from the topsoil layer.
    """
    top = vg_data.iloc[0]
    smfcf = calculate_soil_moisture_content(default_pf_field_capacity(), top.alpha, top.n, top.theta_r, top.theta_s)
    smw = calculate_soil_moisture_content(default_pf_wilting_point(), top.alpha, top.n, top.theta_r, top.theta_s)
    return {"awc": float(smfcf - smw), "bulk_density": float(top.D)}
