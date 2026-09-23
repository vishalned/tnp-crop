import datetime
import os
from typing import Optional, Union

import pandas as pd
import rootutils
from pcse.base import WeatherDataProvider
from pcse.base.weather import WeatherDataContainer
from pcse.input import OpenMeteoWeatherDataProvider

from src.data_pipeline.weather.utils_weather.default_weather_variables import (
    default_openmeteo_model,
    default_weather_start_date,
    default_weather_variables,
)

'''
Columns returned by OpenMeteo -
'LAT', 'LON', 'ELEV', 'IRRAD', 'TMIN', 'TMAX', 'VAP', 'RAIN', 'E0', 'ES0', 'ET0', 'WIND', 'TEMP', 'DAY'
'''

_root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
DEFAULT_WEATHER_CACHE_DIR = os.path.join(str(_root), "data", "raw", "weather")


def request_openmeteo_weather(
    latitude: float,
    longitude: float,
    start_date: Optional[Union[str, datetime.date]] = None,
    openmeteo_model: Optional[str] = None,
    force_update: bool = False,
) -> OpenMeteoWeatherDataProvider:
    """Fetch a PCSE `OpenMeteoWeatherDataProvider` for a location, covering
    `start_date` up to the present (minus the model's publication delay).

    Note the constructor keyword is `openmeteo_model`, not `model` -
    verified against the installed pcse package (v6), since the two differ.

    Prefer `get_weather_provider_for_location`, which caches per exact
    location: PCSE's own cache is keyed only on the location truncated to
    0.1 degree, ignores `start_date` on a hit, so it can silently return a
    date range that doesn't cover what was asked for.
    """
    model = openmeteo_model if openmeteo_model is not None else default_openmeteo_model()
    print(f"getting {model} weather for longitude: {longitude} and latitude: {latitude}")

    wdp = OpenMeteoWeatherDataProvider(
        latitude=latitude,
        longitude=longitude,
        openmeteo_model=model,
        start_date=start_date,
        force_update=force_update,
    )
    return wdp


class CachedWeatherDataProvider(WeatherDataProvider):
    """A PCSE weather provider rebuilt from records previously exported from
    an `OpenMeteoWeatherDataProvider` (see `get_weather_provider_for_location`).
    """

    def __init__(self, records: list, latitude: float, longitude: float, description: str):
        WeatherDataProvider.__init__(self)
        self.latitude = latitude
        self.longitude = longitude
        self.elevation = records[0]["ELEV"]
        self.description = [description]
        for rec in records:
            wdc = WeatherDataContainer(**rec)
            self._store_WeatherDataContainer(wdc, wdc.DAY)


def _weather_cache_path(longitude: float, latitude: float, model: str, cache_dir: str) -> str:
    return os.path.join(cache_dir, f"weather_{longitude}_{latitude}_{model}.csv")


def _load_weather_cache(path: str) -> list:
    df = pd.read_csv(path)
    df["DAY"] = pd.to_datetime(df["DAY"]).dt.date
    records = []
    for rec in df.to_dict(orient="records"):
        records.append({k: v for k, v in rec.items() if not (isinstance(v, float) and pd.isna(v))})
    return records


def get_weather_provider_for_location(
    latitude: float,
    longitude: float,
    start_date: Union[str, datetime.date],
    end_date: Optional[Union[str, datetime.date]] = None,
    openmeteo_model: Optional[str] = None,
    cache_dir: Optional[str] = None,
    force_refresh: bool = False,
) -> WeatherDataProvider:
    """Weather provider for a location covering at least [start_date, end_date],
    downloaded at most once per location.

    The full period from `default_weather_start_date()` (or `start_date`, if
    earlier) to the present is fetched in one request and saved to
    `data/raw/weather/weather_{lon}_{lat}_{model}.csv`, keyed on the exact
    coordinates. Later calls for any year at the same location are served
    from that file. It is re-fetched only if it doesn't cover the requested
    range (or `force_refresh=True`), and always with `force_update=True` so
    PCSE's own coarse 0.1-degree cache can't hand back a different range.

    `end_date` is clipped to the latest date the model publishes, so a
    growing season that runs past the present still hits the cache.
    """
    model = openmeteo_model if openmeteo_model is not None else default_openmeteo_model()
    cache_dir = cache_dir if cache_dir is not None else DEFAULT_WEATHER_CACHE_DIR
    if isinstance(start_date, str):
        start_date = datetime.date.fromisoformat(start_date)
    if isinstance(end_date, str):
        end_date = datetime.date.fromisoformat(end_date)

    latest_available = datetime.date.today() - datetime.timedelta(
        days=OpenMeteoWeatherDataProvider.delay_historical_models[model]
    )
    needed_end = min(end_date, latest_available) if end_date is not None else latest_available

    path = _weather_cache_path(longitude, latitude, model, cache_dir)
    description = f"OpenMeteo {model} weather for lon={longitude}, lat={latitude} (cached at {path})"

    cached_start = None
    if os.path.exists(path) and not force_refresh:
        records = _load_weather_cache(path)
        cached_start, cached_end = records[0]["DAY"], records[-1]["DAY"]
        if cached_start <= start_date and cached_end >= needed_end:
            print(f"Weather cache hit for longitude: {longitude}, latitude: {latitude} ({path}).")
            return CachedWeatherDataProvider(records, latitude, longitude, description)

    fetch_start = min(start_date, default_weather_start_date())
    if cached_start is not None:
        fetch_start = min(fetch_start, cached_start)
    wdp = request_openmeteo_weather(
        latitude=latitude,
        longitude=longitude,
        start_date=fetch_start,
        openmeteo_model=model,
        force_update=True,
    )

    records = sorted(wdp.export(), key=lambda r: r["DAY"])
    os.makedirs(cache_dir, exist_ok=True)
    pd.DataFrame.from_records(records).to_csv(path, index=False)
    print(f"Weather for longitude: {longitude}, latitude: {latitude} cached at {path}.")
    return CachedWeatherDataProvider(records, latitude, longitude, description)


def weather_provider_to_dataframe(
    weather_data_provider: WeatherDataProvider,
    latitude: float,
    longitude: float,
    start_date: Optional[Union[str, datetime.date]] = None,
    end_date: Optional[Union[str, datetime.date]] = None,
) -> pd.DataFrame:
    """Turn an already-fetched weather provider into a tidy DataFrame, one
    row per day, restricted to `default_weather_variables()` and optionally
    clipped to [start_date, end_date]. Lets callers that already hold a
    provider (e.g. the WOFOST runner, which needs the same daily records the
    simulation consumed) avoid fetching it again.
    """
    records = weather_data_provider.export()
    if not records:
        raise ValueError(
            f"OpenMeteo returned no weather records for longitude: {longitude}, latitude: {latitude}."
        )

    df_weather = pd.DataFrame.from_records(records)
    keep_cols = [c for c in default_weather_variables() if c in df_weather.columns]
    df_weather = df_weather[keep_cols].rename(columns={"DAY": "day"})
    df_weather.insert(0, "longitude", longitude)
    df_weather.insert(0, "latitude", latitude)

    if start_date is not None:
        if isinstance(start_date, str):
            start_date = datetime.date.fromisoformat(start_date)
        df_weather = df_weather[df_weather["day"] >= start_date]
    if end_date is not None:
        if isinstance(end_date, str):
            end_date = datetime.date.fromisoformat(end_date)
        df_weather = df_weather[df_weather["day"] <= end_date]

    df_weather = df_weather.sort_values("day").reset_index(drop=True)
    return df_weather


def get_df_weather(
    latitude: float,
    longitude: float,
    start_date: Union[str, datetime.date],
    end_date: Union[str, datetime.date],
    openmeteo_model: Optional[str] = None,
) -> pd.DataFrame:
    """Daily weather for a location as a tidy DataFrame, one row per day,
    restricted to `default_weather_variables()` and clipped to
    [start_date, end_date]. Served from the per-location cache.
    """
    wdp = get_weather_provider_for_location(
        latitude=latitude,
        longitude=longitude,
        start_date=start_date,
        end_date=end_date,
        openmeteo_model=openmeteo_model,
    )
    return weather_provider_to_dataframe(wdp, latitude, longitude, start_date, end_date)
