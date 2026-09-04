from typing import Optional

import pandas as pd
import yaml
from pcse.base import ParameterProvider
from pcse.input import DummySoilDataProvider, WOFOST81SiteDataProvider_Classic, YAMLCropDataProvider

from src.data_pipeline.wofost.utils_wofost.default_wofost_variables import default_site_parameters


def load_soil_data(soil_yaml_path: str) -> dict:
    """Load a PCSE-format soil YAML (e.g. produced by
    `src.data_pipeline.soil.generate_soilgrids_soil_file`) as a plain dict.
    PCSE's `SoilProfile` reads `SoilProfileDescription` straight out of the
    merged `ParameterProvider`, so no dedicated soil-file reader class is
    needed here -- just the parsed YAML.
    """
    with open(soil_yaml_path) as f:
        return yaml.safe_load(f)


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
    soil_yaml_path: Optional[str],
    site_parameters: Optional[dict] = None,
) -> ParameterProvider:
    """Assemble crop + soil + site parameters into one `ParameterProvider`.

    `soil_yaml_path=None` uses PCSE's `DummySoilDataProvider`, appropriate for
    potential-production (phenology-track) runs that don't touch the water
    balance at all (plan step 9).
    """
    crop_data = load_crop_data_provider(model_class, crop_name, variety_name)
    soil_data = load_soil_data(soil_yaml_path) if soil_yaml_path is not None else DummySoilDataProvider()
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
    one array per day, one value per soil layer. Expand those into
    `{column}_layer{i}` scalar columns so the CSV is plain tabular data.
    """
    for column in list(df.columns):
        if len(df) > 0 and hasattr(df[column].iloc[0], "__len__") and not isinstance(df[column].iloc[0], str):
            num_layers = len(df[column].iloc[0])
            for i in range(num_layers):
                df[f"{column}_layer{i}"] = df[column].apply(lambda arr: arr[i])
            df = df.drop(columns=[column])
    return df
