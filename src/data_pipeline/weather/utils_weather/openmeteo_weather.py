import datetime
from typing import Optional, Union

import pandas as pd
from pcse.input import OpenMeteoWeatherDataProvider

from src.data_pipeline.weather.utils_weather.default_weather_variables import (
    default_weather_variables,
    default_openmeteo_model,
)

'''
Columns returned by OpenMeteo - 
'LAT', 'LON', 'ELEV', 'IRRAD', 'TMIN', 'TMAX', 'VAP', 'RAIN', 'E0', 'ES0', 'ET0', 'WIND', 'TEMP', 'DAY'
'''


def request_openmeteo_weather(
    latitude: float,
    longitude: float,
    start_date: Optional[Union[str, datetime.date]] = None,
    openmeteo_model: Optional[str] = None,
    force_update: bool = False,
) -> OpenMeteoWeatherDataProvider:
    """Fetch a PCSE `OpenMeteoWeatherDataProvider` for a location.

    Note the constructor keyword is `openmeteo_model`, not `model` -
    verified against the installed pcse package (v6), since the two differ.
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


def weather_provider_to_dataframe(
    weather_data_provider: OpenMeteoWeatherDataProvider,
    latitude: float,
    longitude: float,
    start_date: Optional[Union[str, datetime.date]] = None,
    end_date: Optional[Union[str, datetime.date]] = None,
) -> pd.DataFrame:
    """Turn an already-fetched `OpenMeteoWeatherDataProvider` into a tidy
    DataFrame, one row per day, restricted to `default_weather_variables()`
    and optionally clipped to [start_date, end_date]. Factored out of
    `get_df_weather` so callers that already hold a provider (e.g. the WOFOST
    runner, which needs the same daily records the simulation consumed) don't
    have to re-fetch it.
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
    """Fetch daily weather for a location and return it as a tidy DataFrame,
    one row per day, restricted to `default_weather_variables()` and clipped
    to [start_date, end_date].
    """
    if isinstance(start_date, str):
        start_date = datetime.date.fromisoformat(start_date)
    if isinstance(end_date, str):
        end_date = datetime.date.fromisoformat(end_date)

    wdp = request_openmeteo_weather(
        latitude=latitude,
        longitude=longitude,
        start_date=start_date,
        openmeteo_model=openmeteo_model,
    )

    return weather_provider_to_dataframe(wdp, latitude, longitude, start_date, end_date)
