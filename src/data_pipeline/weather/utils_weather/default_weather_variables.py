def default_weather_variables():
    """PCSE `WeatherDataContainer` fields to keep from the OpenMeteo pull.

    Covers the daily meteo variables the WOFOST pretraining plan, the Phase 1
    ICL model spec, and the CYBench `meteo` feature group need: min/max/avg
    temperature, precipitation, solar radiation, wind speed, vapour pressure
    (humidity proxy), and reference evapotranspiration (`ET0`, used to derive
    `cwb = RAIN - ET0`).
    """
    return ["DAY", "TMIN", "TMAX", "TEMP", "RAIN", "IRRAD", "WIND", "VAP", "ET0"]


def default_openmeteo_model():
    """ERA5-Land reanalysis, chosen over `best_match` for a deterministic,
    reproducible weather source (see wofost_synthetic_pretraining_plan)."""
    return "era5_land"
