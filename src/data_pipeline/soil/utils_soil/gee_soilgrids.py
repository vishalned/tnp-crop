import pandas as pd

from src.data_pipeline.soil.utils_soil.default_soil_variables import default_zs


def ensure_ee_initialized() -> None:
    """Initialize the Earth Engine client if it isn't already (idempotent).
    Assumes `ee.Authenticate()` has already been run once, outside this
    codebase, per the user's own GEE project setup.
    """
    import ee

    try:
        ee.Number(1).getInfo()
    except Exception:
        ee.Initialize()


def get_df_soilgrids_gee(cfg, longitude: float, latitude: float) -> pd.DataFrame:
    """GEE-based counterpart to `soilgrids.get_df_soilgrids()`: returns the
    same depth-indexed DataFrame shape (`latitude`, `longitude`, `zmin`,
    `zmax` + SoilGrids variables in conventional units) that
    `calculate_van_genuchten` expects, sourced from Earth Engine instead of
    the ISRIC REST API. Unit scaling already happened inside
    `gee_soil_extractor.soil()`. Requires the `earthengine-api` package and
    an authenticated GEE project.
    """
    import ee

    from src.data_pipeline.soil.gee_soil_extractor import soil as gee_soil_extract

    ensure_ee_initialized()
    point = ee.Geometry.Point([longitude, latitude])
    soil_result = gee_soil_extract(cfg, point)

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
