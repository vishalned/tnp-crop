def default_crop_variety():
    """One default cultivar per crop for v1 (wofost_synthetic_pretraining_plan
    decision #16: "don't randomize cultivar yet"). Verified against the actual
    WOFOST_crop_parameters repo (wofost81 branch) variety lists.

    TODO: replace with per-location/region cultivar selection, or cultivar
    randomization for the calibration mechanism in phase1consolidatedsummary
    (both explicitly out of scope for v1).
    """
    return {
        "wheat": "Winter_wheat_101",
        "maize": "Grain_maize_201",
    }


def default_sowing_doy():
    """Placeholder sowing-day-of-year per crop (day of year, 1-indexed),
    standing in until the real WorldCereal start-of-season (SOS) extraction
    from wofost_synthetic_pretraining_plan step 2/5 is wired in.

    TEMP VALUES -- rough NW-Europe sowing windows, not location-aware.
    TODO: replace with a per-location WorldCereal SOS lookup.
    """
    return {
        "wheat": 288,  # ~mid October: typical NW-Europe winter wheat sowing
        "maize": 121,  # ~end of April: typical NW-Europe grain maize sowing
    }


def default_max_duration_days():
    """Upper bound on days from sowing to forced crop-cycle termination. PCSE
    stops the run at this point even if maturity (DVS=2) is never reached --
    exactly the degenerate case step 7 of the plan says to watch for.
    """
    return {
        "wheat": 365,
        "maize": 240,
    }


def default_soil_parameters():
    """v1 soil-step correction: use one generic soil profile for every
    location instead of deriving a per-location root-zone bucket.

    The SoilGrids pull this pipeline already has (clay/nitrogen/phh2o/soc at
    0-5/5-15/15-30cm) is not sufficient input for a water-retention
    pedotransfer function on its own -- no sand/silt fraction (so texture
    isn't actually determined by clay alone) and no bulk density -- and the
    SoilGrids API is currently unreachable anyway. Rather than fetch more
    data or do an unreliable estimate, every location/run uses this same
    fixed bucket for `Wofost81_WLP_CWB`'s classic waterbalance.

    Values are the standard PCSE-tutorial "generic medium soil" numbers (the
    same ones PCSE's own `DummySoilDataProvider` uses for potential-production
    runs where soil doesn't matter) -- not measured, not location-specific.

    TODO: per-location soil hydraulics deferred until sand/silt fraction and
    bulk density are added to the SoilGrids pull (or the API is reachable
    again) -- see `pcse_runner.derive_static_soil_features` for how the
    static `awc`/`bulk_density` features are populated in the meantime.
    """
    return {
        "SMFCF": 0.30,
        "SM0": 0.40,
        "SMW": 0.10,
        "CRAIRC": 0.06,
        "SOPE": 10.0,
        "KSUB": 10.0,
        "RDMSOL": 120.0,
    }


def default_bulk_density():
    """Fixed literature-typical bulk density for a medium-textured/loam
    soil (g/cm^3), used as a placeholder for CYBench's `soil` feature group
    until real per-location bulk density is available.

    TEMP VALUE -- not measured, same for every location/run.
    """
    return 1.35


def default_site_parameters():
    """Site parameters required by WOFOST81SiteDataProvider_Classic that
    generic soil/weather inputs don't otherwise cover.

    TEMP VALUES: WAV (initial soil moisture) and NAVAILI are generic
    placeholders (NAVAILI is unused by the no-N WLP_CWB config, but the site
    data provider still requires a value); CO2 is a fixed present-day default
    rather than a per-year historical value.
    TODO: revisit WAV per soil type if runs turn out sensitive to initial
    moisture; consider a per-year CO2 series for multi-decade runs.
    """
    return {"WAV": 50.0, "CO2": 360.0, "NAVAILI": 80.0}


def wofost_model_name():
    """Water-limited production, classic (single-bucket) waterbalance, no
    nitrogen/SNOMIN, no multi-layer soil -- verified against
    `dir(pcse.models)` (v1 simplification spec). This is the model used to
    generate real training episodes for the CYBench yield task.
    """
    return "Wofost81_WLP_CWB"


def plumbing_check_model_name():
    """Potential production: no soil/water balance at all, so it's the
    quickest way to validate weather -> crop params -> agromanagement ->
    PCSE run -> output extraction end to end (spec step 1). Its output is
    NOT used as real pretraining data -- soil-driven yield variation is the
    whole point of the yield task, and PP has none.
    """
    return "Wofost81_PP"
