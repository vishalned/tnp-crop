import math
from typing import Optional, Tuple

import pandas as pd
import yaml
from pcse.base import ParameterProvider
from pcse.input import DummySoilDataProvider, WOFOST81SiteDataProvider_Classic, YAMLCropDataProvider
from pcse.util import Afgen

from src.data_pipeline.soil.generate_gee_soil_file import generate_soil_file_from_gee
from src.data_pipeline.weather.utils_weather.openmeteo_weather import weather_provider_to_dataframe
from src.data_pipeline.wofost.utils_wofost.default_wofost_variables import (
    default_crop_parameters_dir,
    default_site_parameters,
)


def build_soil_data(longitude: float, latitude: float) -> Tuple[dict, dict]:
    """Fetch a per-location, depth-resolved SoilGrids profile via Earth
    Engine and assemble it into the PCSE multi-layer soil YAML
    `Wofost81_WLP_MLWB` needs (`generate_gee_soil_file.generate_soil_file_from_gee`
    -- van Genuchten curves tabulated per depth, no collapsing to a single
    bucket). Requires the `earthengine-api` package and an authenticated GEE
    project.

    Returns `(soil_data, static_features)`: the parsed `SoilProfileDescription`
    dict ready for `ParameterProvider`, and the topsoil-derived `awc`/
    `bulk_density` CYBench-aligned static features (the multi-layer profile
    doesn't otherwise expose single scalars for the whole soil column).
    """
    result = generate_soil_file_from_gee(longitude=longitude, latitude=latitude)
    with open(result["path"]) as f:
        soil_data = yaml.safe_load(f)
    return soil_data, result["static_features"]


def load_crop_data_provider(model_class, crop_name: str, variety_name: str) -> YAMLCropDataProvider:
    """Load crop parameters from a local clone of the WOFOST_crop_parameters
    repo (see `default_wofost_variables.default_crop_parameters_dir` for the
    clone command and why this fork/branch specifically). No network access
    needed at run time.
    """
    crop_data = YAMLCropDataProvider(fpath=default_crop_parameters_dir())
    # Fetching directly from the upstream ajwdewit GitHub repo instead
    # (network access, decision #16) is also possible:
    #   crop_data = YAMLCropDataProvider(model_class)
    # but that version doesn't include parameters for C4 crops (maize).
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

    params = ParameterProvider(sitedata=site_data, soildata=soil_data, cropdata=crop_data)
    _override_rooting_depth_if_needed(params, soil_data)
    return params


def _override_rooting_depth_if_needed(params: ParameterProvider, soil_data) -> None:
    """Work around a real integration gap between the SoilGrids-based soil
    generator and WOFOST's multilayer waterbalance: `SoilProfile` requires the
    crop's max rootable depth (`RDMCR`, used as-is -- *not* clamped against
    `RDMSOL` -- see `MultiLayerWaterBalance._setup_new_crop`) to exactly
    coincide with a `SoilLayers` cumulative-thickness boundary (see
    `pcse.soil.soil_profile.SoilProfile.validate_max_rooting_depth`), but
    `default_zs()` (0/5/15/30/60/100/200 cm) generally won't include the
    crop's default RDMCR (e.g. 125 cm for Winter_wheat_101).

    TEMP FIX: clamp RDMCR down to the deepest available soil layer boundary
    at or below its default value, via PCSE's parameter-override mechanism,
    rather than failing the run. TODO: align the soil generator's depth bins
    with common crop rooting depths (or vice versa) instead of overriding.
    """
    if not isinstance(soil_data, dict) or "SoilProfileDescription" not in soil_data:
        return  # DummySoilDataProvider (potential production) has no layers to align to

    layers = soil_data["SoilProfileDescription"]["SoilLayers"]
    boundaries = []
    cumulative_depth = 0.0
    for layer in layers:
        cumulative_depth += layer["Thickness"]
        boundaries.append(cumulative_depth)

    max_rootable_depth = params["RDMCR"]
    aligned_boundaries = [b for b in boundaries if b <= max_rootable_depth]
    if not aligned_boundaries:
        target_depth = boundaries[-1]
    elif max_rootable_depth in aligned_boundaries:
        return  # already aligned, nothing to override
    else:
        target_depth = aligned_boundaries[-1]

    params.set_override("RDMCR", target_depth)


def run_wofost(model_class, params: ParameterProvider, weather_data_provider, agromanagement: list):
    """Run a PCSE/WOFOST engine to completion and return the daily driver +
    output trajectory as a DataFrame, plus the terminal summary dict.
    """
    engine = model_class(params, weather_data_provider, agromanagement)
    engine.run_till_terminate()

    daily_output = pd.DataFrame(engine.get_output())
    daily_output = _flatten_per_layer_columns(daily_output)
    summary_output = engine.get_summary_output()
    summary = summary_output[0] if summary_output else {}

    return daily_output, summary


def _flatten_per_layer_columns(df: pd.DataFrame) -> pd.DataFrame:
    """The multilayer waterbalance reports some outputs (e.g. `SM`, `WC`) as
    one array per day, one value per soil layer (shallowest first). Expand
    those into `{column}_layer{i}` scalar columns so the CSV is plain
    tabular data.
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
    - `ssm` = WOFOST's simulated topsoil `SM` (`SM_layer0`, the shallowest
      multi-layer waterbalance layer) -- closest available match to
      CYBench's satellite-derived surface soil moisture
    """
    weather_df = weather_provider_to_dataframe(weather_data_provider, latitude, longitude)
    weather_cols = ["day", "TMIN", "TMAX", "TEMP", "RAIN", "IRRAD", "ET0"]
    merged = daily_output.merge(weather_df[weather_cols], on="day", how="left")

    merged["cwb"] = merged["RAIN"] - merged["ET0"]

    k_diftb = Afgen(params["KDIFTB"])
    merged["fpar"] = merged.apply(lambda row: 1.0 - math.exp(-k_diftb(row["DVS"]) * row["LAI"]), axis=1)

    merged["ssm"] = merged["SM_layer0"]

    return merged
