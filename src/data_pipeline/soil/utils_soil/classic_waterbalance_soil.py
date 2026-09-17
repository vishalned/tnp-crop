import pandas as pd

from src.data_pipeline.soil.utils_soil.default_soil_variables import (
    default_pf_field_capacity,
    default_pf_wilting_point,
    default_soilgrid_d_factors,
    default_zs,
)
from src.data_pipeline.soil.utils_soil.generate_soil_files import (
    calculate_van_genuchten,
    calculate_soil_moisture_content,
)

# Critical air content: fixed default, matches the constant already hardcoded
# in generate_soil_files.generate_df_soil_input for the multi-layer profile.
DEFAULT_CRAIRC = 0.03


def gee_soil_result_to_dataframe(soil_result: dict, latitude: float, longitude: float) -> pd.DataFrame:
    """Convert the GEE `soil()` extractor's output
    (`{"modality", "data", "variable_names"}`, one row per depth layer,
    one column per variable) into the depth-indexed DataFrame
    `calculate_van_genuchten()` expects: `latitude`/`longitude`/`zmin`/`zmax`
    plus the SoilGrids variables in real physical units.

    SoilGrids band values served via Earth Engine's `sample()` are the raw
    mapped integers (e.g. clay in g/kg * 10); unlike the SoilGrids REST API
    (which reports a `d_factor` per response that `get_df_soilgrids` divides
    by dynamically), GEE doesn't hand back that metadata, so the same fixed,
    ISRIC-documented conversion factors are applied here explicitly (see
    `default_soilgrid_d_factors`).
    """
    df_raw = soil_result["data"]
    zmins, zmaxs = default_zs()
    if len(df_raw) != len(zmins):
        raise ValueError(
            f"Expected {len(zmins)} depth layers (matching default_zs()), got {len(df_raw)} "
            "from the GEE soil extraction -- check the depth_layers config."
        )

    d_factors = default_soilgrid_d_factors()
    df = pd.DataFrame({"latitude": latitude, "longitude": longitude, "zmin": zmins, "zmax": zmaxs})
    for var in df_raw.columns:
        df[var] = df_raw[var].to_numpy() / d_factors.get(var, 1)

    return df


def collapse_to_root_zone_bucket(df_soilgrids: pd.DataFrame, rooting_depth_cm: float) -> dict:
    """Collapse a multi-depth SoilGrids profile into the single root-zone
    bucket `Wofost81_WLP_CWB`'s classic waterbalance needs (SMFCF, SM0, SMW,
    CRAIRC, SOPE, KSUB, RDMSOL), instead of the full per-layer
    `SoilProfileDescription` used for the multi-layer waterbalance.

    Per the v1 simplification spec: a rooting-depth-weighted average of the
    van Genuchten parameters across whatever SoilGrids depths fall within
    [0, rooting_depth_cm], not a depth-resolved profile. Van Genuchten
    parameters don't strictly average linearly, but this is an accepted v1
    approximation -- revisit if runs turn out sensitive to it.

    Also returns `BULK_DENSITY`, the same rooting-depth-weighted average
    applied to `bdod` -- not a WOFOST model parameter, just carried through
    for `pcse_runner.derive_static_soil_features` to expose as the
    CYBench-matching `bulk_density` static feature.
    """
    df_vgp = calculate_van_genuchten(df_soilgrids)

    weights = []
    for zmin, zmax in zip(df_vgp.zmin, df_vgp.zmax):
        overlap = max(0.0, min(zmax, rooting_depth_cm) - min(zmin, rooting_depth_cm))
        weights.append(overlap)
    weights = pd.Series(weights, index=df_vgp.index)
    if weights.sum() <= 0:
        raise ValueError(
            f"No SoilGrids depth layer overlaps the rooting depth {rooting_depth_cm} cm; "
            "cannot build a root-zone soil bucket."
        )
    weights = weights / weights.sum()

    alpha = float((df_vgp.alpha * weights).sum())
    n = float((df_vgp.n * weights).sum())
    theta_r = float((df_vgp.theta_r * weights).sum())
    theta_s = float((df_vgp.theta_s * weights).sum())
    k_sat = float((df_vgp.k_sat * weights).sum())
    bulk_density = float((df_soilgrids.set_index(df_vgp.index).bdod * weights).sum())

    smfcf = calculate_soil_moisture_content(default_pf_field_capacity(), alpha, n, theta_r, theta_s)
    smw = calculate_soil_moisture_content(default_pf_wilting_point(), alpha, n, theta_r, theta_s)

    return {
        "SMFCF": smfcf,
        "SM0": theta_s,
        "SMW": smw,
        "CRAIRC": DEFAULT_CRAIRC,
        # TODO: SOPE (max percolation rate, root zone) and KSUB (max percolation
        # rate, subsoil) both set to the root-zone saturated conductivity as a
        # v1 placeholder -- revisit if subsoil drainage should differ.
        "SOPE": k_sat,
        "KSUB": k_sat,
        "RDMSOL": rooting_depth_cm,
        "BULK_DENSITY": bulk_density,
    }
