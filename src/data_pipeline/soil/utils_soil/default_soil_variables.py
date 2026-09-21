def default_soilgrid_variables():
    # Define variables that need to be collected for this location
    soil_variables = ["bdod", "clay", "phh2o", "sand", "silt", "soc", "nitrogen"]
    return soil_variables


def default_zs():
    # Define minimum and maximum depths for each SoilGrids soil layer
    zmins = [0, 5, 15, 30, 60, 100]
    zmaxs = [5, 15, 30, 60, 100, 200]
    return zmins, zmaxs


def default_soilgrid_d_factors():
    """Fixed ISRIC SoilGrids v2.0 conversion factors: raw mapped value /
    factor = value in the variable's target physical unit.

    The SoilGrids REST API (`soilgrids.py`) reports this per response as
    `unit_measure.d_factor` and `get_df_soilgrids` divides by it dynamically.
    Earth Engine's `sample()` doesn't return that metadata, only the raw
    mapped integers -- these are the same constants ISRIC publishes for the
    product, used to bring GEE-sourced values into the same physical units.
    """
    return {
        "bdod": 100,      # cg/cm3 -> kg/dm3 (g/cm3)
        "clay": 10,       # g/kg -> g/100g (%)
        "nitrogen": 100,  # cg/kg -> g/kg
        "phh2o": 10,      # pH*10 -> pH
        "sand": 10,       # g/kg -> g/100g (%)
        "silt": 10,       # g/kg -> g/100g (%)
        "soc": 10,        # dg/kg -> g/kg
    }


def default_som_content():
    """
    Default soil organic matter content, it is assumed to be 58%
    """

    return 0.58


def default_range_pf_values():
    return [-1.0, 1.0, 1.3, 1.7, 2.0, 2.3, 2.4, 2.7, 3.0, 3.3, 3.7, 4.0, 4.2, 6.0]


def default_pf_field_capacity():
    return 2.0


def default_pf_wilting_point():
    return 4.2


def default_surface_conductivity():
    return 70


# --------------------------------------------------------------------------
# GEE-based SoilGrids extraction config
# (src.data_pipeline.soil.utils_soil.gee_soilgrids.soil / get_df_soilgrids_gee)
#
# Replaces the standalone configs/data_pipeline/soil/gee_soil.yaml Hydra
# file -- nothing else ever read it, so it's simpler kept as a plain default
# here alongside the rest of this module, matching default_soilgrid_variables()
# above for the REST-based path.
# --------------------------------------------------------------------------

def default_gee_soil_variables():
    """All 7 SoilGrids variables the GEE `SOIL_ASSETS` map supports.
    `nitrogen`/`phh2o` aren't used by the van Genuchten hydraulic derivation
    (`calculate_van_genuchten`) but are kept available as potential extra
    static features.
    """
    return ["clay", "nitrogen", "phh2o", "soc", "bdod", "sand", "silt"]


def default_gee_depth_layers():
    """SoilGrids depth-layer strings (e.g. "0-5cm") for the GEE band-name
    convention (`f"{variable}_{depth}_mean"`), built from `default_zs()` so
    there's one source of truth for the depth grid.
    """
    zmins, zmaxs = default_zs()
    return [f"{zmin}-{zmax}cm" for zmin, zmax in zip(zmins, zmaxs)]


def default_gee_soil_config(name: str = "soil", log_level: str = "INFO") -> dict:
    """Default config for the GEE `soil()` extractor / `get_df_soilgrids_gee()`:
    all 7 SoilGrids variables, the full 6-depth grid. Going past 30cm matters
    -- wheat/maize root zones extend well beyond it, and the multi-layer
    waterbalance needs one `SoilLayers` entry per depth.
    """
    return {
        "name": name,
        "log_level": log_level,
        "variables": default_gee_soil_variables(),
        "depth_layers": default_gee_depth_layers(),
    }
