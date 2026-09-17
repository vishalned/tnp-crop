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
