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
| `data/raw/weather/weather_{lon}_{lat}_{model}.csv` | All daily weather for a location, from `default_weather_start_date()` (2000-01-01) to the present, in one file | **Cached**: downloaded once per location and reused for every year simulated there. It is re-fetched only if a run needs dates it doesn't cover, in which case it's extended back to the requested start. Fetches bypass PCSE's own `~/.pcse/meteo_cache`, which is keyed on a 0.1°-truncated location and ignores the requested start date. |
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

### Weather drivers (joined in from the OpenMeteo pull, not PCSE output)

| Column | Meaning | Unit |
|---|---|---|
| `TMIN` / `TMAX` / `TEMP` | Daily min / max / average air temperature | °C |
| `RAIN` | Daily precipitation | cm/day |
| `IRRAD` | Daily global radiation | J/m²/day |
| `ET0` | Reference evapotranspiration (computed by the OpenMeteo weather provider) | cm/day |

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
