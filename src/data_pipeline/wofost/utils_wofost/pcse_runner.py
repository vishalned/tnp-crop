import copy
import math
from typing import Optional

import pandas as pd
from pcse.base import ParameterProvider
from pcse.input import DummySoilDataProvider, WOFOST81SiteDataProvider_Classic, YAMLCropDataProvider
from pcse.util import Afgen

from src.data_pipeline.weather.utils_weather.openmeteo_weather import weather_provider_to_dataframe
from src.data_pipeline.wofost.utils_wofost.default_wofost_variables import (
    default_crop_parameters_dir,
    default_site_parameters,
)


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
    param_overrides: Optional[dict] = None,
) -> ParameterProvider:
    """Assemble crop + soil + site parameters into one `ParameterProvider`.

    `soil_data=None` uses PCSE's `DummySoilDataProvider`, appropriate only for
    the potential-production plumbing check (spec step 1) that doesn't touch
    the water balance at all -- never for a real yield-track episode.

    `param_overrides` (name -> value) replace crop parameters, e.g. a sampled
    world's scaled ones. If they set `RDMCR`, the soil layer that depth falls
    in is split in two identical layers at it (`split_soil_layer_at`), so the
    multi-layer water balance accepts it as is; without overrides the
    crop's own RDMCR is rounded down to a layer boundary as before.
    """
    param_overrides = param_overrides or {}
    crop_data = load_crop_data_provider(model_class, crop_name, variety_name)
    if "RDMCR" in param_overrides and isinstance(soil_data, dict):
        soil_data = split_soil_layer_at(soil_data, param_overrides["RDMCR"])
    soil_data = soil_data if soil_data is not None else DummySoilDataProvider()
    site_params = site_parameters if site_parameters is not None else default_site_parameters()
    site_data = WOFOST81SiteDataProvider_Classic(**site_params)

    params = ParameterProvider(sitedata=site_data, soildata=soil_data, cropdata=crop_data)
    for name, value in param_overrides.items():
        params.set_override(name, value)
    if getattr(model_class, "__name__", "") == "Wofost81_PP" and isinstance(soil_data, dict) and "SMFCF" not in params:
        # potential production (Wofost81_PP) uses the classic soil bucket, which keeps soil moisture at
        # field capacity (growth isn't water-limited) but needs its scalar parameters: the topsoil's
        for name, value in topsoil_classic_parameters(soil_data).items():
            params.set_override(name, value, check=False)
    _override_rooting_depth_if_needed(params, soil_data)
    return params


def topsoil_classic_parameters(soil_data: dict) -> dict:
    """The classic (single-bucket) soil parameters from the top layer of a multi-layer description:
    water content at saturation (SM0, lowest pF of the table), field capacity (SMFCF) and wilting
    point (SMW) from its SMfromPF curve, and its critical air content (CRAIRC)."""
    profile = soil_data["SoilProfileDescription"]
    top = profile["SoilLayers"][0]
    sm = Afgen(top["SMfromPF"])
    return {"SM0": float(sm(top["SMfromPF"][0])), "SMFCF": float(sm(profile.get("PFFieldCapacity", 2.0))),
            "SMW": float(sm(profile.get("PFWiltingPoint", 4.2))), "CRAIRC": float(top["CRAIRC"])}


def layer_thicknesses(soil_data: dict) -> list:
    return [layer["Thickness"] for layer in soil_data["SoilProfileDescription"]["SoilLayers"]]


def merge_split_layers(daily: pd.DataFrame, split: list, original: list) -> pd.DataFrame:
    """Undo `split_soil_layer_at` in a daily output: the per-layer columns of a profile with layer
    thicknesses `split` back to the `original` layers (water amounts `WC_layer*` add up, volumetric
    contents `SM_layer*` are thickness-weighted), so layer i is the same depth in every output."""
    if split == original:
        return daily
    groups, j = [], 0
    for thickness in original:
        members, total = [], 0.0
        while total < thickness - 1e-9:
            members.append(j)
            total += split[j]
            j += 1
        groups.append(members)
    daily = daily.copy()
    for name, weighted in (("WC", False), ("SM", True)):
        if f"{name}_layer0" not in daily:
            continue
        merged = {}
        for i, members in enumerate(groups):
            cols = [daily[f"{name}_layer{m}"] for m in members]
            if weighted:
                merged[f"{name}_layer{i}"] = sum(c * split[m] for c, m in zip(cols, members)) / original[i]
            else:
                merged[f"{name}_layer{i}"] = sum(cols)
        daily = daily.drop(columns=[f"{name}_layer{m}" for m in range(len(split))]).assign(**merged)
    return daily


def split_soil_layer_at(soil_data: dict, depth_cm: float) -> dict:
    """A copy of a PCSE multi-layer soil description in which the layer that
    `depth_cm` falls inside is split in two layers with the same properties,
    so `depth_cm` becomes a layer boundary (the multi-layer water balance
    needs the max rooting depth on one). Unchanged if it already is one or
    lies outside the profile."""
    soil_data = copy.deepcopy(soil_data)
    layers = soil_data["SoilProfileDescription"]["SoilLayers"]
    top = 0.0
    for i, layer in enumerate(layers):
        bottom = top + layer["Thickness"]
        if top < depth_cm < bottom:
            upper, lower = copy.deepcopy(layer), copy.deepcopy(layer)
            upper["Thickness"] = depth_cm - top
            lower["Thickness"] = bottom - depth_cm
            layers[i:i + 1] = [upper, lower]
            break
        top = bottom
    return soil_data


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

    When the campaign starts before sowing or runs past maturity (see
    `agromanagement.build_agromanagement`), the crop columns (DVS, LAI, ...)
    are empty on those days while the soil columns (SM, WC) are simulated.
    """
    weather_df = weather_provider_to_dataframe(weather_data_provider, latitude, longitude)
    weather_cols = ["day", "TMIN", "TMAX", "TEMP", "RAIN", "IRRAD", "ET0"]
    merged = daily_output.merge(weather_df[weather_cols], on="day", how="left")

    merged["cwb"] = merged["RAIN"] - merged["ET0"]

    k_diftb = Afgen(params["KDIFTB"])
    dvs = pd.to_numeric(merged["DVS"], errors="coerce")
    lai = pd.to_numeric(merged["LAI"], errors="coerce")
    # No crop before sowing / after maturity (DVS, LAI missing): bare field, fpar 0.
    merged["fpar"] = [
        1.0 - math.exp(-k_diftb(d) * l) if pd.notna(d) and pd.notna(l) else 0.0 for d, l in zip(dvs, lai)
    ]

    merged["ssm"] = merged["SM_layer0"]

    return merged
