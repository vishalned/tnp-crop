def default_weather_variables():
    """PCSE `WeatherDataContainer` fields to keep from the OpenMeteo pull.

    Covers the daily meteo variables both the WOFOST pretraining plan and the
    Phase 1 ICL model spec need as context: min/max temperature, precipitation,
    solar radiation, wind speed, and vapour pressure (humidity proxy).
    """
    return ["DAY", "TMIN", "TMAX", "RAIN", "IRRAD", "WIND", "VAP"]


def default_openmeteo_model():
    """ERA5-Land reanalysis, chosen over `best_match` for a deterministic,
    reproducible weather source (see wofost_synthetic_pretraining_plan)."""
    return "era5_land"
