import os

import rootutils

_root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def default_crop_parameters_dir():
    """Local clone of the WOFOST_crop_parameters repo, wofost81 branch, from
    the herman-berghuijs fork rather than the upstream ajwdewit one -- this
    fork includes a fix for C4 crops (maize) that upstream doesn't have yet.

    Set up once via:
        git clone -b wofost81 https://github.com/herman-berghuijs/WOFOST_crop_parameters.git data/crop_parameters/wofost81

    See README.md's "Crop parameters" section.
    """
    return os.path.join(str(_root), "data", "crop_parameters", "wofost81")


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


def default_site_parameters():
    """Site parameters required by WOFOST81SiteDataProvider_Classic that
    generic soil/weather inputs don't otherwise cover.

    TEMP VALUES: WAV (initial soil moisture) and NAVAILI are generic
    placeholders (NAVAILI is unused by the no-N MLWB config, but the site
    data provider still requires a value); CO2 is a fixed present-day default
    rather than a per-year historical value.
    TODO: revisit WAV per soil type if runs turn out sensitive to initial
    moisture; consider a per-year CO2 series for multi-decade runs.
    """
    return {"WAV": 50.0, "CO2": 360.0, "NAVAILI": 80.0}


def wofost_model_name():
    """Water-limited production, multi-layer waterbalance, no nitrogen/SNOMIN
    -- verified against `dir(pcse.models)`. Now that the full 6-depth GEE
    SoilGrids pull provides real depth-resolved van Genuchten curves, there's
    no reason to collapse them into a single CWB bucket; MLWB consumes each
    layer directly. This is the model used to generate real training
    episodes for the CYBench yield task.
    """
    return "Wofost81_WLP_MLWB"


def plumbing_check_model_name():
    """Potential production: no soil/water balance at all, so it's the
    quickest way to validate weather -> crop params -> agromanagement ->
    PCSE run -> output extraction end to end (spec step 1). Its output is
    NOT used as real pretraining data -- soil-driven yield variation is the
    whole point of the yield task, and PP has none.
    """
    return "Wofost81_PP"
