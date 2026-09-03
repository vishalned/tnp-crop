def default_soilgrid_variables():
    # Define variables that need to be collected for this location
    soil_variables = ["bdod", "clay", "phh2o", "sand", "silt", "soc", "nitrogen"]
    return soil_variables


def default_zs():
    # Define minimum and maximum depths for each SoilGrids soil layer
    zmins = [0, 5, 15, 30, 60, 100]
    zmaxs = [5, 15, 30, 60, 100, 200]
    return zmins, zmaxs


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
