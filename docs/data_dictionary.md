# Data dictionary

Reference for the columns/fields produced by the data pipeline
(`src/data_pipeline/`), in particular the WOFOST run outputs
(`data/raw/wofost/{crop}/wofost_{crop}_{lon}_{lat}_{year}_{sowing_date}.csv`
and the matching `..._summary.json`).

## Where things get saved under `data/raw/`

| Path | What | Cached / re-fetched? |
|---|---|---|
| `data/raw/soilgrids_gee/soil_{lon}_{lat}.yaml` | Multi-layer PCSE soil profile for a location | **Cached** — reused if it already exists (soil only depends on location; GEE quota is the scarce resource). Pass `force_refresh=True` / `--force-refresh-soil` to bypass. |
| `data/raw/soilgrids_gee/soil_{lon}_{lat}_static_features.json` | Sidecar with the topsoil `awc`/`bulk_density` for that same location | Cached alongside the YAML above. |
| `data/raw/weather/weather_{lon}_{lat}_gee_era5_land.csv` | Daily ERA5-Land weather for a location (Earth Engine `ECMWF/ERA5_LAND/DAILY_AGGR`), already in PCSE units incl. `E0`/`ES0`/`ET0`, for whichever calendar years have been needed so far (not necessarily contiguous). **Active source** (`gee_weather.get_gee_weather_provider_for_location`). | **Cached**, one file per location: a run fetches only the calendar years its season spans that aren't in the file yet (e.g. 2005 + 2006 for a season sown Oct 2005) and merges them in. Unpublished recent days (ERA5-Land lags a few months) are re-checked at most every 7 days. |
| `~/.pcse/meteo_cache/OpenMeteoWeatherDataProvider_LAT..._LON..._{model}.cache` | Only if you use the Open-Meteo path (`openmeteo_weather.request_openmeteo_weather`, no longer called by the pipeline). PCSE's own cache: data from the requested `start_date` to the present. | Managed by PCSE: keyed on the location truncated to 0.1° + model, reused for 90 days **regardless of `start_date`** — so request the earliest start first for a location. |
| `data/raw/wofost/{crop}/wofost_{crop}_{lon}_{lat}_{year}_{sowing_date}.csv` + `..._summary.json` | One simulation episode's daily trajectory + summary | Always freshly written, one pair per episode. |
| `data/raw/wofost/dataset_manifest.csv` | One row per episode attempted by `generate_wofost_dataset.py`, `status`/`error` plus the summary fields and file paths | Written incrementally by the batch runner; not itself a cache. |

Model: [`Wofost81_WLP_MLWB`](https://pcse.readthedocs.io) — water-limited
production, multi-layer waterbalance, no nitrogen/SNOMIN (see
`default_wofost_variables.wofost_model_name()`).

## Daily trajectory CSV

### Crop state (from PCSE's own `Wofost81` crop module)

| Column | Meaning | Unit |
|---|---|---|
| `day` | Simulation date | date |
| `DVS` | Development stage: <0 before emergence, 0 = emergence, 1 = anthesis/flowering, 2 = maturity | – |
| `LAI` | Leaf area index (leaf area per unit ground area) | m² leaf / m² ground |
| `TAGP` | Total above-ground production (dry matter) | kg/ha |
| `WSO` | Weight of storage organs (grain/tuber) — this is the yield component | kg/ha |
| `WLV` | Weight of living leaves | kg/ha |
| `WST` | Weight of living stems | kg/ha |
| `WRT` | Weight of living roots | kg/ha |
| `TRA` | Actual crop transpiration rate | cm/day |
| `RD` | Current rooting depth | cm |

### Water balance (multi-layer; `_layer0` = shallowest / topsoil, increasing with depth)

| Column | Meaning | Unit |
|---|---|---|
| `WWLOW` | Total water in the profile (rooted + unrooted zone) | cm |
| `RFTRA` | Reduction factor on transpiration due to water stress (1 = no stress, 0 = fully stressed) | – |
| `SM_layer{i}` | Volumetric soil moisture content of layer `i` | cm³/cm³ |
| `WC_layer{i}` | Water content (depth-equivalent) of layer `i` | cm |

### Nitrogen bookkeeping — present but not a real N-limitation model

`Wofost81_WLP_MLWB` uses PCSE's `Wofost81` crop module, which *always* runs
an internal N-demand/uptake sub-model regardless of the water/nutrient
config chosen — see `WOFOST81SiteDataProvider_Classic`'s `NAVAILI`
requirement. For this config, the soil side wires in PCSE's
`N_PotentialProduction` stub (not a real soil N balance), whose entire
implementation is: *`NAVAIL` stays fixed at 100 kg/ha for the whole run,
whatever the crop takes* — it does **not** read the `NAVAILI` site parameter
we set (that parameter is only used by the more advanced N-limited configs,
e.g. `SoilModuleWrapper_NWLP_MLWB_CNB`). So these columns reflect an
always-unconstrained N economy, not simulated soil nitrogen availability:

| Column | Meaning | Unit |
|---|---|---|
| `NAVAIL` | "Available" soil N — hardcoded to 100 for this config, never simulated | kg N/ha |
| `Ndemand` | Crop's N demand for the day | kg N/ha/day |
| `RNuptake` | Rate of N uptake by the crop | kg N/ha/day |
| `NuptakeTotal` | Cumulative N taken up so far | kg N/ha |
| `NamountSO`/`NamountLV`/`NamountST`/`NamountRT` | N content of storage organs / leaves / stems / roots | kg N/ha |

### Weather drivers (joined in from the GEE ERA5-Land pull, not PCSE output)

| Column | Meaning | Unit |
|---|---|---|
| `TMIN` / `TMAX` / `TEMP` | Daily min / max / average air temperature | °C |
| `RAIN` | Daily precipitation | cm/day |
| `IRRAD` | Daily global radiation | J/m²/day |
| `ET0` | Reference evapotranspiration, Penman-Monteith via PCSE's `reference_ET` (computed in `gee_weather.era5_land_to_pcse_records`) | cm/day |

### Derived features (CYBench-feature-set-aligned; see `pcse_runner.merge_weather_and_derive_features`)

| Column | Meaning |
|---|---|
| `cwb` | Climatic water balance = `RAIN - ET0` |
| `fpar` | Fraction of absorbed photosynthetically active radiation, `1 - exp(-k(DVS) * LAI)`, with `k` read from the crop's own `KDIFTB` (light-extinction) table |
| `ssm` | Surface soil moisture proxy = `SM_layer0` (topsoil layer) — closest available match to CYBench's satellite-derived surface soil moisture |

## Summary JSON (per run)

| Field | Meaning |
|---|---|
| `longitude` / `latitude` | Location |
| `crop` | `wheat` or `maize` |
| `variety_name` | Cultivar used (see `default_wofost_variables.default_crop_variety`) |
| `year` | Simulation year |
| `sowing_date` | Actual (jittered) sowing date used |
| `yield_kg_per_ha` | `TWSO` — total weight of storage organs at maturity (the yield label) |
| `final_dvs` | `DVS` at the end of the run (2.0 = reached maturity; less than that = run terminated early, a degenerate case worth checking) |
| `awc` | Available water capacity of the topsoil layer, computed manually as `SMFCF - SMW` from the van Genuchten water-retention curve (SoilGrids has no direct field for this) |
| `bulk_density` | Topsoil bulk density, read directly from SoilGrids' `bdod` (`bdod_mean` asset) — not computed |

## Soil YAML (`SoilProfileDescription`, one file per location)

Produced by `generate_gee_soil_file.py` from the GEE SoilGrids pull, via
`calculate_van_genuchten` → `generate_df_soil_input` → `generate_soil_yaml`
(all in `utils_soil/generate_soil_files.py`).

| Field | Meaning |
|---|---|
| `RDMSOL` | Maximum soil depth (200 cm — deepest SoilGrids layer) |
| `PFFieldCapacity` / `PFWiltingPoint` | pF (log10 of soil water tension) at field capacity (2.0) and wilting point (4.2) |
| `SurfaceConductivity` | Saturated hydraulic conductivity of the soil surface |
| `SoilLayers[i].Thickness` | Layer thickness (cm), from `default_zs()` |
| `SoilLayers[i].RHOD` | Bulk density (`bdod`, scaled) |
| `SoilLayers[i].Soil_pH` | `phh2o` |
| `SoilLayers[i].FSOMI` | Organic matter fraction, derived from `soc` |
| `SoilLayers[i].CNRatioSOMI` | Carbon:nitrogen ratio, `soc / nitrogen` |
| `SoilLayers[i].CRAIRC` | Critical air content (fixed default) |
| `SoilLayers[i].SMfromPF` | Soil moisture content tabulated at each pF value in `default_range_pf_values()` — the van Genuchten retention curve |
| `SoilLayers[i].CONDfromPF` | log10 hydraulic conductivity tabulated at the same pF values — the Mualem-van Genuchten conductivity curve |
| `SubSoilType` | Properties below the deepest downloaded depth (200cm) — simplifying assumption: reuses the deepest layer's own properties |
| `GroundWater` | `false` — no water table modeling |

## SoilGrids raw variables (fetched, not all consumed by the hydraulic derivation)

| Variable | Used for | Unit after scaling |
|---|---|---|
| `clay`, `sand`, `silt` | Texture → van Genuchten pedotransfer function | % |
| `bdod` | Bulk density (`RHOD`, and the `bulk_density` static feature) | g/cm³ |
| `soc` | Organic matter fraction (`FSOMI`) | g/kg |
| `phh2o` | Soil pH (`Soil_pH`) | pH |
| `nitrogen` | `CNRatioSOMI` (`soc / nitrogen`) | g/kg |

See `default_soilgrid_d_factors()` for the raw-SoilGrids-integer → these
units conversion factors.

## Processed training table (`data/processed/wofost_{crop}_daily.csv`)

Built by `src/data_pipeline/wofost/process_wofost_dataset.py` from a batch
run's `dataset_manifest.csv` (successful episodes only), the episodes'
daily CSVs and the per-location weather caches. One file per crop, plus a
`..._columns.json` sidecar listing the column groups.

**One row = one simulated growing season** (sowing -> maturity) at one
location, the same unit as a CYBench row (region x season). `year` is the
**harvest year** (CYBench convention): winter wheat sown Oct 2013 and
harvested Jul 2014 has `year = 2014`, `sowing_year = 2013`.

| Column(s) | Meaning |
|---|---|
| `sample_id`, `location_index`, `location_id`, `crop`, `year`, `sowing_year` | Identifiers (`location_id` = `"{lon}_{lat}"`) |
| `latitude`, `longitude`, `awc`, `bulk_density` | Static features (soil ones from the topsoil, as in the summary JSON) |
| `sos_doy` | Nominal start-of-season day of year for the crop (`default_sowing_doy()`, stand-in for the WorldCereal SOS) |
| `sowing_doy`, `sowing_date` | Actual (jittered) sowing day |
| `maturity_date`, `season_length_days`, `reached_maturity` | When DVS reached 2 (metadata — not known at forecast time, don't use as an input feature) |
| `window_start` | Date of day 0 of the time series |
| `yield_t_per_ha` / `yield_kg_per_ha` | **Target**: `TWSO` (storage-organ dry matter) at maturity |
| `{feature}_d{k:03d}` | Daily time series (unaggregated), day `k` counted from `window_start` |

Day 0 is `--pre-season-days` (default 30) before the anchor: the crop's
nominal season start that year (`--align season_start`, default) or the
actual sowing date (`--align sowing`). The anchor is at index
`anchor_day_index` in the columns JSON. Every row of a crop has
`pre_season_days + max_duration` days (wheat: 30 + 365 = 395), independent
of when the crop matured.

| Feature | Source | Unit |
|---|---|---|
| `tmin`, `tmax`, `tavg` | weather cache | °C |
| `prec` | weather cache (`RAIN`) | mm/day |
| `rad` | weather cache (`IRRAD`) | MJ/m²/day |
| `et0` | weather cache (`ET0`) | mm/day |
| `cwb` | `prec - et0` | mm/day |
| `fpar` | daily WOFOST CSV | fraction; 0 before sowing and after maturity (bare field) |
| `ssm` | daily WOFOST CSV (`SM_layer0`) | cm³/cm³; NaN before sowing and after maturity (not simulated) |

Note: WOFOST's `TWSO` is dry matter, while reported (e.g. CYBench) yields
are usually at a standard moisture content, so absolute levels differ.
