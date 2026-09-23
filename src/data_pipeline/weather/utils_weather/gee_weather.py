"""ERA5-Land daily weather from Google Earth Engine, as a PCSE weather provider.

Drop-in alternative to `openmeteo_weather.get_weather_provider_for_location`
that avoids Open-Meteo's free-tier rate limits (HTTP 429). The Open-Meteo
module is left untouched; this one sits alongside it.

Source: `ECMWF/ERA5_LAND/DAILY_AGGR` -- the same ERA5-Land reanalysis
Open-Meteo's `era5_land` model serves, pre-aggregated to daily values by
Google (days are UTC days).

Request pattern (the efficient part): the whole period for a location
(`default_weather_start_date()` to the latest available day) comes back in a
single `ImageCollection.getRegion` call -- one round trip per location, not
one per year or per variable. ~26 years x 8 bands is about 110k values,
well under getRegion's 1,048,576-value limit; longer periods are split into
as few chunks as that limit allows. Elevation is added as an extra band on
every image, so it rides along in the same request rather than costing a
separate one. The result is cached per location in
`data/raw/weather/weather_{lon}_{lat}_gee_era5_land.csv` and every
year/sowing date simulated there is cut from that file.

Conversions mirror PCSE's `OpenMeteoWeatherDataProvider._prepare_weather_dataframe`
so both sources produce the same PCSE record layout
(DAY, LAT, LON, ELEV, TMIN, TMAX, TEMP, IRRAD, RAIN, WIND, VAP, E0, ES0, ET0).
"""

import datetime
import logging
import os
from typing import Optional, Union

import numpy as np
import pandas as pd
import rootutils
from pcse.base import WeatherDataProvider
from pcse.exceptions import PCSEError
from pcse.util import check_angstromAB, reference_ET, wind10to2

from src.data_pipeline.soil.utils_soil.gee_soilgrids import ensure_ee_initialized
from src.data_pipeline.weather.utils_weather.default_weather_variables import default_weather_start_date
from src.data_pipeline.weather.utils_weather.openmeteo_weather import (
    CachedWeatherDataProvider,
    _load_weather_cache,
    weather_provider_to_dataframe,
)

log = logging.getLogger(__name__)

_root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
DEFAULT_WEATHER_CACHE_DIR = os.path.join(str(_root), "data", "raw", "weather")

ERA5_LAND_DAILY_COLLECTION = "ECMWF/ERA5_LAND/DAILY_AGGR"
ERA5_LAND_SCALE_M = 11132  # native ~0.1 degree grid
ERA5_LAND_BANDS = [
    "temperature_2m",  # K, daily mean
    "temperature_2m_min",  # K
    "temperature_2m_max",  # K
    "dewpoint_temperature_2m",  # K, daily mean
    "total_precipitation_sum",  # m/day
    "surface_solar_radiation_downwards_sum",  # J/m2/day
    "u_component_of_wind_10m",  # m/s, daily mean
    "v_component_of_wind_10m",  # m/s, daily mean
]
# Global DEM (GMTED2010 mean elevation, 7.5 arc-sec); only used for the
# psychrometric constant in the ET0 calculation, so exact matching to the
# ERA5-Land orography isn't important.
ELEVATION_ASSET = "USGS/GMTED2010_FULL"
ELEVATION_BAND = "mea"
GETREGION_MAX_VALUES = 1_048_576

# PCSE's Open-Meteo provider defaults, used when Angstrom A/B can't be
# estimated from the data (fewer than 200 days, or out-of-range estimates).
DEFAULT_ANGSTROM_A = 0.29
DEFAULT_ANGSTROM_B = 0.49
ET_MODEL = "PM"

# How long a cache that stops before a requested end date is trusted before
# asking GEE again for newer days. ERA5-Land is published with a lag of a few
# months, so a season that runs past the latest published day would otherwise
# trigger a re-fetch on every call.
CACHE_END_RECHECK_DAYS = 7

GEE_WEATHER_SOURCE = "gee_era5_land"


def _import_ee():
    # Local import so the rest of the data pipeline still works without
    # earthengine-api installed; only the GEE-based paths need it.
    import ee

    return ee


def _weather_cache_path(longitude: float, latitude: float, cache_dir: str) -> str:
    return os.path.join(cache_dir, f"weather_{longitude}_{latitude}_{GEE_WEATHER_SOURCE}.csv")


def request_gee_era5_land_raw(
    latitude: float,
    longitude: float,
    start_date: datetime.date,
    end_date: datetime.date,
) -> pd.DataFrame:
    """Raw ERA5-Land daily bands (+ elevation) for one point over
    [start_date, end_date], via as few `getRegion` calls as the value limit
    allows (one for any realistic period). Returns one row per day with a
    `DAY` column and the raw band values in their native units.
    """
    ensure_ee_initialized()
    ee = _import_ee()

    point = ee.Geometry.Point([longitude, latitude])
    elevation = ee.Image(ELEVATION_ASSET).select([ELEVATION_BAND], ["elevation"])
    bands = ERA5_LAND_BANDS + ["elevation"]

    # getRegion returns id, longitude, latitude, time + one value per band.
    values_per_day = len(bands) + 4
    max_days = GETREGION_MAX_VALUES // values_per_day - 1

    frames = []
    chunk_start = start_date
    while chunk_start <= end_date:
        chunk_end = min(end_date, chunk_start + datetime.timedelta(days=max_days - 1))
        collection = (
            ee.ImageCollection(ERA5_LAND_DAILY_COLLECTION)
            # filterDate's end is exclusive
            .filterDate(chunk_start.isoformat(), (chunk_end + datetime.timedelta(days=1)).isoformat())
            .select(ERA5_LAND_BANDS)
            .map(lambda img: img.addBands(elevation))
        )
        print(
            f"getting GEE ERA5-Land weather for longitude: {longitude}, latitude: {latitude}, "
            f"{chunk_start} to {chunk_end}"
        )
        region = collection.getRegion(point, ERA5_LAND_SCALE_M).getInfo()
        header, rows = region[0], region[1:]
        if rows:
            frames.append(pd.DataFrame(rows, columns=header))
        chunk_start = chunk_end + datetime.timedelta(days=1)

    if not frames:
        raise ValueError(
            f"GEE returned no ERA5-Land weather for longitude: {longitude}, latitude: {latitude} "
            f"between {start_date} and {end_date}."
        )

    df = pd.concat(frames, ignore_index=True)
    df["DAY"] = pd.to_datetime(df["time"], unit="ms", utc=True).dt.date
    df = df[["DAY"] + bands].drop_duplicates("DAY").sort_values("DAY").reset_index(drop=True)

    # Points over sea / outside the ERA5-Land land mask come back as nulls.
    n_before = len(df)
    df = df.dropna(subset=ERA5_LAND_BANDS)
    if df.empty:
        raise ValueError(
            f"ERA5-Land has no data at longitude: {longitude}, latitude: {latitude} "
            "(point is probably outside the ERA5-Land land mask)."
        )
    if len(df) < n_before:
        log.warning(
            "Dropped %d days with missing ERA5-Land values at lon=%s, lat=%s.",
            n_before - len(df), longitude, latitude,
        )
    return df


def _toa_radiation_mj(day_of_year: np.ndarray, latitude: float) -> np.ndarray:
    """Daily top-of-atmosphere radiation (MJ/m2/day), FAO-56; same formula as
    PCSE's `OpenMeteoWeatherDataProvider.calculate_toa_radiation`."""
    g_sc = 1361
    d_r = 1 + 0.033 * np.cos(2 * np.pi * day_of_year / 365)
    delta = np.radians(23.45 * np.sin(2 * np.pi * (day_of_year - 81) / 365))
    phi = np.radians(latitude)
    with np.errstate(invalid="ignore"):
        h_s = np.arccos(-np.tan(phi) * np.tan(delta))
    h0 = (24 * 3600 / np.pi) * g_sc * d_r * (
        np.cos(phi) * np.cos(delta) * np.sin(h_s) + h_s * np.sin(phi) * np.sin(delta)
    )
    return h0 / 1e6


def _estimate_angstrom_ab(df: pd.DataFrame, latitude: float) -> tuple:
    """Angstrom A/B from the 5th / 98th percentile of IRRAD / TOA radiation,
    as PCSE's Open-Meteo and NASA POWER providers do."""
    if len(df) < 200:
        return DEFAULT_ANGSTROM_A, DEFAULT_ANGSTROM_B

    doys = pd.to_datetime(df["DAY"]).dt.dayofyear.to_numpy()
    relative_radiation = (df["IRRAD"].to_numpy() / 1e6) / _toa_radiation_mj(doys, latitude)
    relative_radiation = relative_radiation[np.isfinite(relative_radiation)]
    if relative_radiation.size < 200:
        return DEFAULT_ANGSTROM_A, DEFAULT_ANGSTROM_B

    angstrom_a = float(np.percentile(relative_radiation, 5))
    angstrom_b = float(np.percentile(relative_radiation, 98)) - angstrom_a
    try:
        check_angstromAB(angstrom_a, angstrom_b)
    except PCSEError as e:
        log.warning("Angstrom A/B (%f, %f) out of range: %s. Using defaults.", angstrom_a, angstrom_b, e)
        return DEFAULT_ANGSTROM_A, DEFAULT_ANGSTROM_B
    return angstrom_a, angstrom_b


def era5_land_to_pcse_records(df_raw: pd.DataFrame, latitude: float, longitude: float) -> list:
    """Convert raw ERA5-Land daily bands to PCSE weather records (units as
    PCSE's `WeatherDataContainer` expects), including E0/ES0/ET0.

    - TMIN/TMAX/TEMP: K -> degC
    - RAIN: m/day -> cm/day
    - IRRAD: already J/m2/day
    - VAP: hPa, from mean dewpoint (Tetens, as PCSE's Open-Meteo provider)
    - WIND: 10 m -> 2 m (log profile). Note: DAILY_AGGR only has daily-mean
      u/v components, so the speed is the magnitude of the mean wind vector,
      which is <= the mean wind speed on days when the direction changes.
      Minor for ET0, but it is a small systematic low bias.
    - E0/ES0/ET0: PCSE's `reference_ET` (Penman / Penman-Monteith), mm -> cm
    """
    elev = df_raw["elevation"].dropna()
    elevation = float(elev.iloc[0]) if not elev.empty else 0.0

    df = pd.DataFrame({"DAY": df_raw["DAY"].to_numpy()})
    df["TMIN"] = df_raw["temperature_2m_min"].to_numpy() - 273.15
    df["TMAX"] = df_raw["temperature_2m_max"].to_numpy() - 273.15
    df["TEMP"] = df_raw["temperature_2m"].to_numpy() - 273.15
    df["RAIN"] = np.clip(df_raw["total_precipitation_sum"].to_numpy(), 0, None) * 100.0
    df["IRRAD"] = np.clip(df_raw["surface_solar_radiation_downwards_sum"].to_numpy(), 0, None)
    tdew = df_raw["dewpoint_temperature_2m"].to_numpy() - 273.15
    df["VAP"] = 6.108 * np.exp((17.27 * tdew) / (tdew + 237.3))
    wind10 = np.sqrt(
        df_raw["u_component_of_wind_10m"].to_numpy() ** 2 + df_raw["v_component_of_wind_10m"].to_numpy() ** 2
    )
    df["WIND"] = wind10to2(wind10)
    df["LAT"] = latitude
    df["LON"] = longitude
    df["ELEV"] = elevation

    angstrom_a, angstrom_b = _estimate_angstrom_ab(df, latitude)

    records = []
    for rec in df.to_dict(orient="records"):
        e0, es0, et0 = reference_ET(
            DAY=rec["DAY"], LAT=latitude, ELEV=elevation,
            TMIN=rec["TMIN"], TMAX=rec["TMAX"], IRRAD=rec["IRRAD"],
            VAP=rec["VAP"], WIND=rec["WIND"],
            ANGSTA=angstrom_a, ANGSTB=angstrom_b, ETMODEL=ET_MODEL,
        )
        rec["E0"], rec["ES0"], rec["ET0"] = e0 / 10.0, es0 / 10.0, et0 / 10.0
        records.append({k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in rec.items()})
    return records


def get_gee_weather_provider_for_location(
    latitude: float,
    longitude: float,
    start_date: Optional[Union[str, datetime.date]] = None,
    end_date: Optional[Union[str, datetime.date]] = None,
    cache_dir: Optional[str] = None,
    force_refresh: bool = False,
) -> WeatherDataProvider:
    """GEE ERA5-Land counterpart of
    `openmeteo_weather.get_weather_provider_for_location`: a PCSE weather
    provider for a location covering at least [start_date, end_date],
    downloaded at most once per location.

    The whole period from `default_weather_start_date()` (or `start_date`,
    if earlier) up to today is requested in one `getRegion` call (GEE simply
    returns up to the latest published day) and saved to
    `data/raw/weather/weather_{lon}_{lat}_gee_era5_land.csv`, keyed only on
    the exact coordinates. Every later call at that location -- any year,
    any sowing date -- is cut from that file.

    It is re-fetched only if:
    - `start_date` is earlier than the cache (extended back, keeping the
      existing start as a lower bound), or
    - `end_date` is later than the cache *and* the cache is older than
      `CACHE_END_RECHECK_DAYS` (new ERA5-Land days may have been published), or
    - `force_refresh=True`.
    """
    cache_dir = cache_dir if cache_dir is not None else DEFAULT_WEATHER_CACHE_DIR
    if start_date is None:
        start_date = default_weather_start_date()
    if isinstance(start_date, str):
        start_date = datetime.date.fromisoformat(start_date)
    if isinstance(end_date, str):
        end_date = datetime.date.fromisoformat(end_date)

    today = datetime.date.today()
    needed_end = min(end_date, today) if end_date is not None else today

    path = _weather_cache_path(longitude, latitude, cache_dir)
    description = f"GEE ERA5-Land daily weather for lon={longitude}, lat={latitude} (cached at {path})"

    cached_start = None
    if os.path.exists(path) and not force_refresh:
        records = _load_weather_cache(path)
        cached_start, cached_end = records[0]["DAY"], records[-1]["DAY"]
        cache_age_days = (
            datetime.datetime.now() - datetime.datetime.fromtimestamp(os.path.getmtime(path))
        ).days
        covers_start = cached_start <= start_date
        covers_end = cached_end >= needed_end or cache_age_days < CACHE_END_RECHECK_DAYS
        if covers_start and covers_end:
            print(f"Weather cache hit for longitude: {longitude}, latitude: {latitude} ({path}).")
            return CachedWeatherDataProvider(records, latitude, longitude, description)

    fetch_start = min(start_date, default_weather_start_date())
    if cached_start is not None:
        fetch_start = min(fetch_start, cached_start)

    df_raw = request_gee_era5_land_raw(latitude, longitude, fetch_start, today)
    records = era5_land_to_pcse_records(df_raw, latitude, longitude)

    os.makedirs(cache_dir, exist_ok=True)
    pd.DataFrame.from_records(records).to_csv(path, index=False)
    print(f"Weather for longitude: {longitude}, latitude: {latitude} cached at {path}.")
    return CachedWeatherDataProvider(records, latitude, longitude, description)


def get_df_weather_gee(
    latitude: float,
    longitude: float,
    start_date: Optional[Union[str, datetime.date]] = None,
    end_date: Optional[Union[str, datetime.date]] = None,
) -> pd.DataFrame:
    """Daily GEE ERA5-Land weather for a location as a tidy DataFrame, one
    row per day, restricted to `default_weather_variables()` and clipped to
    [start_date, end_date]. Served from the per-location cache.
    """
    wdp = get_gee_weather_provider_for_location(
        latitude=latitude,
        longitude=longitude,
        start_date=start_date,
        end_date=end_date,
    )
    return weather_provider_to_dataframe(wdp, latitude, longitude, start_date, end_date)
