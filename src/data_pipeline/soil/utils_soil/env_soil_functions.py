import os
import joblib
import yaml
import numpy as np

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

_SCENARIO_PATH = os.path.join(str(root), "configs", "data_pipeline", "scenarios")

try:
    from src.data_pipeline.soil.utils_soil.soil_feature_extractor import extract_soil_features
except ImportError:
    from default_soil_variables import extract_soil_features

# Cache of models and transformed soils
_scalers, _pcas = {}, {}


def load_region_models(region: str):
    if region not in _scalers:
        if region in ['zeeland', 'groningen', 'gelderland']:
            _scalers[region] = joblib.load(os.path.join(_SCENARIO_PATH, f"{region}", f"soil_scaler_{region}.joblib"))
            _pcas[region] = joblib.load(os.path.join(_SCENARIO_PATH, f"{region}", f"soil_pca_{region}.joblib"))
        else:
            _scalers[region] = joblib.load(os.path.join(_SCENARIO_PATH, "soil_scaler_all.joblib"))
            _pcas[region] = joblib.load(os.path.join(_SCENARIO_PATH, "soil_pca_all.joblib"))
    return _scalers[region], _pcas[region]


def soil_to_latent_pca(soil_params: dict, region: str) -> np.ndarray:
    """Transform soil to PCA latent for its region."""
    scaler, pca = load_region_models(region)
    _, vec, _ = extract_soil_features(soil_params)
    z = pca.transform(scaler.transform(vec[None, :]))[0].astype(np.float32)
    return z
