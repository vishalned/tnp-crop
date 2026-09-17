import math
from typing import Optional

import pandas as pd
from pcse.base import ParameterProvider
from pcse.input import DummySoilDataProvider, WOFOST81SiteDataProvider_Classic, YAMLCropDataProvider
from pcse.util import Afgen

from src.data_pipeline.soil.utils_soil.classic_waterbalance_soil import collapse_to_root_zone_bucket
from src.data_pipeline.soil.utils_soil.soilgrids import get_df_soilgrids
from src.data_pipeline.weather.utils_weather.openmeteo_weather import weather_provider_to_dataframe
from src.data_pipeline.wofost.utils_wofost.default_wofost_variables import default_site_parameters


def build_soil_data(longitude: float, latitude: float, rooting_depth_cm: float) -> dict:
    """Fetch a raw SoilGrids profile for a location and collapse it into the
    single root-zone bucket `Wofost81_WLP_CWB` needs (v1 simplification spec:
    classic waterbalance, not the multi-layer profile). Requires network
    access to the SoilGrids REST API.
    """
    df_soilgrids = get_df_soilgrids(lat=latitude, lon=longitude)
    return collapse_to_root_zone_bucket(df_soilgrids, rooting_depth_cm=rooting_depth_cm)


def load_crop_data_provider(model_class, crop_name: str, variety_name: str) -> YAMLCropDataProvider:
    """Fetch crop parameters from the WOFOST_crop_parameters GitHub repository
    (decision #16), using the branch matching `model_class` (e.g. wofost81 for
    Wofost81_* models). Requires network access.
    """
    crop_data = YAMLCropDataProvider(model_class)
    crop_data.set_active_crop(crop_name, variety_name)
    return crop_data


def build_parameter_provider(
    model_class,
    crop_name: str,
    variety_name: str,
    soil_data: Optional[dict],
    site_parameters: Optional[dict] = None,
) -> ParameterProvider:
    """Assemble crop + soil + site parameters into one `ParameterProvider`.

    `soil_data=None` uses PCSE's `DummySoilDataProvider`, appropriate only for
    the potential-production plumbing check (spec step 1) that doesn't touch
    the water balance at all -- never for a real yield-track episode.
    """
    crop_data = load_crop_data_provider(model_class, crop_name, variety_name)
    soil_data = soil_data if soil_data is not None else DummySoilDataProvider()
    site_params = site_parameters if site_parameters is not None else default_site_parameters()
    site_data = WOFOST81SiteDataProvider_Classic(**site_params)

    return ParameterProvider(sitedata=site_data, soildata=soil_data, cropdata=crop_data)


def run_wofost(model_class, params: ParameterProvider, weather_data_provider, agromanagement: list):
    """Run a PCSE/WOFOST engine to completion and return the daily state/rate
    output as a DataFrame, plus the terminal summary dict.
    """
    engine = model_class(params, weather_data_provider, agromanagement)
    engine.run_till_terminate()

    daily_output = pd.DataFrame(engine.get_output())
    daily_output = _flatten_per_layer_columns(daily_output)
    summary_output = engine.get_summary_output()
    summary = summary_output[0] if summary_output else {}

    return daily_output, summary


def _flatten_per_layer_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Defensive no-op for the classic (single-bucket) waterbalance, which
    reports scalars: kept in case an output ever comes back as one array per
    day (as the multi-layer waterbalance's `SM`/`WC` do), so it gets expanded
    into `{column}_layer{i}` scalar columns instead of leaking a raw array
    into the CSV.
    """
    for column in list(df.columns):
        if len(df) > 0 and hasattr(df[column].iloc[0], "__len__") and not isinstance(df[column].iloc[0], str):
            num_layers = len(df[column].iloc[0])
            for i in range(num_layers):
                df[f"{column}_layer{i}"] = df[column].apply(lambda arr: arr[i])
            df = df.drop(columns=[column])
    return df


def merge_weather_and_derive_features(
    daily_output: pd.DataFrame,
    weather_data_provider,
    latitude: float,
    longitude: float,
    params: ParameterProvider,
) -> pd.DataFrame:
    """Join the WOFOST daily output with the daily weather drivers it
    consumed and add the v1 CYBench-aligned derived features (spec's "per-run
    output to store"):

    - `cwb` = `RAIN - ET0` (climatic water balance)
    - `fpar` = `1 - exp(-k(DVS) * LAI)`, with `k` read from the crop's own
      `KDIFTB` (extinction coefficient for diffuse light) table
    - `ssm` = WOFOST's simulated `SM` (soil moisture) state -- a straight
      rename, since the classic waterbalance already reports one root-zone
      value rather than CYBench's satellite-derived surface value
    """
    weather_df = weather_provider_to_dataframe(weather_data_provider, latitude, longitude)
    weather_cols = ["day", "TMIN", "TMAX", "TEMP", "RAIN", "IRRAD", "ET0"]
    merged = daily_output.merge(weather_df[weather_cols], on="day", how="left")

    merged["cwb"] = merged["RAIN"] - merged["ET0"]

    k_diftb = Afgen(params["KDIFTB"])
    merged["fpar"] = merged.apply(lambda row: 1.0 - math.exp(-k_diftb(row["DVS"]) * row["LAI"]), axis=1)

    merged["ssm"] = merged["SM"]

    return merged


def derive_static_soil_features(soil_data: dict) -> dict:
    """Static soil features for the v1 CYBench-aligned feature set: `awc`
    (available water capacity) as `SMFCF - SMW`. `bulk_density` is skipped in
    this version (would need a WISE, or fallback SoilGrids, pull -- out of
    scope for now).
    """
    return {"awc": soil_data["SMFCF"] - soil_data["SMW"]}
