from collections import OrderedDict
from typing import Dict, Any, List, Tuple
import argparse
import math
import numpy as np
import os
import yaml
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import re
import matplotlib.pyplot as plt
from collections import defaultdict
import joblib

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

_CONFIG_PATH = os.path.join(str(root), "configs", "data_pipeline")
_SOIL_PATH = os.path.join(_CONFIG_PATH, "soil")
_SOILGRIDS_PATH = os.path.join(_SOIL_PATH, "soilgrids")
_SCENARIO_PATH = os.path.join(_CONFIG_PATH, "scenarios")

ROUND_DECIMALS = 4



# ---------------------- config / choices ----------------------
# pF support where we will interpolate for compact yet meaningful features
PF_TARGETS = np.array([-1.0, 1.0, 2.0, 3.0, 4.2, 6.0], dtype=float)
# which summary stats to compute (thickness-weighted across layers)
STATS = ("mean_w", "min", "max", "std")

# ---------------------- helpers ----------------------
def _as_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return default

def _pairs_from_flat(lst: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Turn [pF1, v1, pF2, v2, ...] into (pf, val) arrays, dedupe and sort by pF."""
    if not isinstance(lst, list) or len(lst) < 2:
        return np.array([]), np.array([])
    it = iter(lst)
    pf_vals, vals = [], []
    for pf, v in zip(it, it):
        pf_vals.append(_as_float(pf))
        vals.append(_as_float(v))
    pf = np.array(pf_vals, dtype=float)
    val = np.array(vals, dtype=float)
    # sort & dedupe by pF
    order = np.argsort(pf)
    pf, val = pf[order], val[order]
    uniq_pf, uniq_idx = np.unique(pf, return_index=True)
    return uniq_pf, val[uniq_idx]

def _interp_at(pf_src: np.ndarray, val_src: np.ndarray, pf_targets: np.ndarray) -> np.ndarray:
    """Safe linear interpolation with edge hold (extrapolate by nearest)."""
    if pf_src.size == 0 or val_src.size == 0:
        return np.full_like(pf_targets, np.nan, dtype=float)
    # np.interp extrapolates by endpoints when left/right are outside range
    return np.interp(pf_targets, pf_src, val_src)

def _thickness_weights(layers: List[Dict[str, Any]]) -> np.ndarray:
    thks = np.array([_as_float(L.get("Thickness", np.nan)) for L in layers], dtype=float)
    thks[np.isnan(thks)] = 0.0
    return thks

def _w_stats(values: np.ndarray, weights: np.ndarray) -> Dict[str, float]:
    """Thickness-weighted mean + unweighted min/max/std; robust to NaNs."""
    out = {}
    v = values.astype(float)
    w = weights.astype(float)
    mask = ~np.isnan(v)
    v, w = v[mask], w[mask]
    if v.size == 0:
        return {"mean_w": np.nan, "min": np.nan, "max": np.nan, "std": np.nan}
    w_sum = w.sum()
    mean_w = (v * w).sum() / w_sum if w_sum > 0 else np.nan
    out["mean_w"] = mean_w
    out["min"] = np.nanmin(v)
    out["max"] = np.nanmax(v)
    out["std"] = np.nanstd(v, ddof=0)
    return out

def _prefix_stats(prefix: str, values: np.ndarray, weights: np.ndarray, feats: OrderedDict):
    stats = _w_stats(values, weights)
    for k, v in stats.items():
        feats[f"{prefix}_{k}"] = float(v)

def _append_vec(features: OrderedDict, names: List[str], vec: List[float], key: str, arr: np.ndarray):
    for i, v in enumerate(arr):
        features[f"{key}[{i}]"] = float(v)
        names.append(f"{key}[{i}]")
        vec.append(float(v))

# ---------------------- main extractor ----------------------
def extract_soil_features(soil: Dict[str, Any]) -> Tuple[OrderedDict, np.ndarray, List[str]]:
    """
    Extract a compact, flat feature vector from a PCSE-like soil calibration dict.

    Returns:
        features (OrderedDict): key -> value for inspection/logging
        vector (np.ndarray): flat numeric vector aligned with names
        names (List[str]): feature names in vector order
    """
    feats = OrderedDict()
    vec: List[float] = []
    names: List[str] = []

    # Top-level convenience handles
    RDMSOL = _as_float(soil.get("RDMSOL", np.nan))
    spd = soil.get("SoilProfileDescription", {}) or {}
    layers: List[Dict[str, Any]] = spd.get("SoilLayers", []) or []
    subsoil = spd.get("SubSoilType", None)

    # Core profile params
    PF_wp = _as_float(spd.get("PFWiltingPoint", np.nan))
    PF_fc = _as_float(spd.get("PFFieldCapacity", np.nan))
    surf_cond = _as_float(spd.get("SurfaceConductivity", np.nan))
    groundwater = 1.0 if bool(spd.get("GroundWater", False)) else 0.0

    base_items = {
        # "RDMSOL_cm": RDMSOL,
        # "PF_wilting": PF_wp,
        # "PF_fieldcap": PF_fc,
        # "SurfaceConductivity": surf_cond,
        # "GroundWater_flag": groundwater,
        "n_layers": float(len(layers)),
        # "has_subsoil": 1.0 if subsoil else 0.0,
    }
    for k, v in base_items.items():
        feats[k] = float(v)
        names.append(k)
        vec.append(float(v))

    if len(layers) == 0:
        return feats, np.array(vec, dtype=float), names

    # Thickness weights
    thk = _thickness_weights(layers)
    thk_sum = thk.sum()
    feats["thickness_total_cm"] = float(thk_sum)
    names.append("thickness_total_cm")
    vec.append(float(thk_sum))

    # Layer scalar properties (thickness-weighted stats)
    for key, nice in [
        ("RHOD", "bulk_density"),
        ("Soil_pH", "pH"),
        ("CNRatioSOMI", "CN"),
        ("CRAIRC", "air_entry"),
        ("FSOMI", "SOM_frac"),
    ]:
        arr = np.array([_as_float(L.get(key, np.nan)) for L in layers], dtype=float)
        _prefix_stats(nice, arr, thk, feats)
        for stat_name in STATS:
            names.append(f"{nice}_{stat_name}")
            vec.append(float(feats[f"{nice}_{stat_name}"]))

    # Interpolate SMfromPF and CONDfromPF per layer to PF_TARGETS
    sm_mat = []   # water content θ at PF_TARGETS per layer
    cond_mat = [] # conductivity (often log10 K) at PF_TARGETS per layer

    for L in layers:
        pf_sm, val_sm = _pairs_from_flat(L.get("SMfromPF", []))
        pf_k,  val_k  = _pairs_from_flat(L.get("CONDfromPF", []))
        sm_i = _interp_at(pf_sm, val_sm, PF_TARGETS)
        k_i  = _interp_at(pf_k,  val_k,  PF_TARGETS)
        sm_mat.append(sm_i)
        cond_mat.append(k_i)

    sm_mat = np.array(sm_mat, dtype=float)   # shape [n_layers, n_pf]
    cond_mat = np.array(cond_mat, dtype=float)

    # Thickness-weighted profile means at each PF target
    # (also handy if you want to feed the whole small curve to a network)
    # We also include min/max across layers (unweighted) at each PF for texture contrast.
    with np.errstate(invalid="ignore"):
        w = thk[:, None]  # broadcast to [n_layers, 1]
        w_sum = np.where(thk_sum > 0, thk_sum, np.nan)
        sm_mean_w = np.nansum(sm_mat * w, axis=0) / w_sum
        sm_min = np.nanmin(sm_mat, axis=0)
        sm_max = np.nanmax(sm_mat, axis=0)

        k_mean_w = np.nansum(cond_mat * w, axis=0) / w_sum
        k_min = np.nanmin(cond_mat, axis=0)
        k_max = np.nanmax(cond_mat, axis=0)

    # Store curves compactly
    _append_vec(feats, names, vec, "theta@pF", sm_mean_w)
    _append_vec(feats, names, vec, "theta_min@pF", sm_min)
    _append_vec(feats, names, vec, "theta_max@pF", sm_max)
    _append_vec(feats, names, vec, "logK@pF", k_mean_w)
    _append_vec(feats, names, vec, "logK_min@pF", k_min)
    _append_vec(feats, names, vec, "logK_max@pF", k_max)

    # Compute plant-available water capacity (PAWC) to RDMSOL:
    # PAWC = sum_over_layers( (theta_FC - theta_WP) * thickness_cm ) * 0.1 [mm]
    # (1 cm water over 1 cm depth = 1 cm * θ; convert cm to mm by *10; but θ is cm3/cm3.)
    def _theta_at(pf_target, mat):
        # thickness-weighted average θ at a single pF
        # (reuse sm_mean_w we already computed if pf_target is in PF_TARGETS)
        if pf_target in PF_TARGETS.tolist():
            idx = PF_TARGETS.tolist().index(pf_target)
            return sm_mean_w[idx]
        # otherwise, compute directly by layer interpolation:
        vals = []
        for L in layers:
            pf_sm, val_sm = _pairs_from_flat(L.get("SMfromPF", []))
            vals.append(_interp_at(pf_sm, val_sm, np.array([pf_target]))[0])
        vals = np.array(vals, dtype=float)
        return np.nansum(vals * thk) / (thk_sum if thk_sum > 0 else np.nan)

    theta_fc = _theta_at(PF_fc if not math.isnan(PF_fc) else 2.0, sm_mat)
    theta_wp = _theta_at(PF_wp if not math.isnan(PF_wp) else 4.2, sm_mat)

    # Also compute PAWC layer-by-layer (more accurate when θ varies with depth)
    # using our interpolated matrices if FC/WP are in the target grid; otherwise re-interp.
    def _layerwise_pawc_mm():
        # get per-layer theta at FC/WP
        def _layer_theta_at(pf_target):
            if pf_target in PF_TARGETS.tolist():
                idx = PF_TARGETS.tolist().index(pf_target)
                return sm_mat[:, idx]
            # else re-interp layerwise:
            tv = []
            for L in layers:
                pf_sm, val_sm = _pairs_from_flat(L.get("SMfromPF", []))
                tv.append(_interp_at(pf_sm, val_sm, np.array([pf_target]))[0])
            return np.array(tv, dtype=float)

        th_fc = _layer_theta_at(PF_fc if not math.isnan(PF_fc) else 2.0)
        th_wp = _layer_theta_at(PF_wp if not math.isnan(PF_wp) else 4.2)
        dtheta = np.clip(th_fc - th_wp, a_min=0.0, a_max=None)  # no negative storage
        # water depth in mm ≈ dtheta * thickness_cm * 10
        pawc_mm = np.nansum(dtheta * thk * 10.0)
        return float(pawc_mm)

    pawc_mm = _layerwise_pawc_mm()
    feats["PAWC_mm_to_RDMSOL"] = float(pawc_mm)
    names.append("PAWC_mm_to_RDMSOL")
    vec.append(float(pawc_mm))

    # Include theta at FC/WP (profile-average) and the delta
    feats["theta_at_FC_mean_w"] = float(theta_fc)
    feats["theta_at_WP_mean_w"] = float(theta_wp)
    feats["theta_FC_minus_WP"] = float(theta_fc - theta_wp)
    names += ["theta_at_FC_mean_w", "theta_at_WP_mean_w", "theta_FC_minus_WP"]
    vec += [float(theta_fc), float(theta_wp), float(theta_fc - theta_wp)]

    # Basic conductivity markers (profile means) at FC and at saturation
    def _logK_at(pf_target):
        if pf_target in PF_TARGETS.tolist():
            idx = PF_TARGETS.tolist().index(pf_target)
            return k_mean_w[idx]
        vals = []
        for L in layers:
            pf_k, val_k = _pairs_from_flat(L.get("CONDfromPF", []))
            vals.append(_interp_at(pf_k, val_k, np.array([pf_target]))[0])
        vals = np.array(vals, dtype=float)
        return np.nansum(vals * thk) / (thk_sum if thk_sum > 0 else np.nan)

    feats["logK_at_sat_pF-1"] = float(_logK_at(-1.0))
    feats["logK_at_FC"] = float(_logK_at(PF_fc if not math.isnan(PF_fc) else 2.0))
    feats["logK_at_WP"] = float(_logK_at(PF_wp if not math.isnan(PF_wp) else 4.2))
    names += ["logK_at_sat_pF-1", "logK_at_FC", "logK_at_WP"]
    vec += [feats["logK_at_sat_pF-1"], feats["logK_at_FC"], feats["logK_at_WP"]]

    # (Optional) subsoil echo features (can help when layers change below RDMSOL)
    if subsoil:
        for k in ["RHOD", "Soil_pH", "CNRatioSOMI", "CRAIRC", "FSOMI", "Thickness"]:
            val = _as_float(subsoil.get(k, np.nan))
            feats[f"subsoil_{k}"] = float(val)
            names.append(f"subsoil_{k}")
            vec.append(float(val))
        # subsoil θ/logK at selected PFs
        pf_sm, val_sm = _pairs_from_flat(subsoil.get("SMfromPF", []))
        pf_k, val_k = _pairs_from_flat(subsoil.get("CONDfromPF", []))
        theta_sub = _interp_at(pf_sm, val_sm, PF_TARGETS)
        logk_sub = _interp_at(pf_k, val_k, PF_TARGETS)
        _append_vec(feats, names, vec, "subsoil_theta@pF", theta_sub)
        _append_vec(feats, names, vec, "subsoil_logK@pF", logk_sub)

    return feats, np.array(vec, dtype=float), names


FNAME_RE = re.compile(r"^soil_([-\d\.]+)_([-\d\.]+)\.yaml$")

def parse_lon_lat_from_fname(fname: str):
    """
    Parse lon, lat from 'soil_{lon}_{lat}.yaml'. Returns floats or (None, None) if no match.
    """
    m = FNAME_RE.match(fname)
    if not m:
        return None, None
    lon = float(m.group(1))
    lat = float(m.group(2))
    return lon, lat

def build_region_lookup(scenario_dict):
    """
    scenario_dict: {'gelderland': [[lon, lat], ...], 'zeeland': [...], ...}
    Returns a dict mapping (round(lon, ROUND_DECIMALS), round(lat, ROUND_DECIMALS)) -> region
    """
    lut = {}
    for region, pairs in scenario_dict.items():
        for pair in pairs:
            # pair format is [lon, lat]
            lon, lat = float(pair[0]), float(pair[1])
            key = (round(lon, ROUND_DECIMALS), round(lat, ROUND_DECIMALS))
            lut[key] = region
    return lut

def region_for_file(fname, region_lut):
    lon, lat = parse_lon_lat_from_fname(fname)
    if lon is None:
        return "unknown"
    key = (round(lon, ROUND_DECIMALS), round(lat, ROUND_DECIMALS))
    return region_lut.get(key, "unknown")


def do_pca():
    # --------------------------------------------------------------------------------------
    # 1) Load scenario coords (region lookup)
    # --------------------------------------------------------------------------------------
    with open(os.path.join(_SCENARIO_PATH, "scenario_coords.yaml"), "r") as f:
        scenario_dict = yaml.safe_load(f)

    region_lut = build_region_lookup(scenario_dict)

    # --------------------------------------------------------------------------------------
    # 2) Load all soil YAMLs
    # --------------------------------------------------------------------------------------
    soil_data = {}
    soil_regions = {}  # filename -> region

    for fname in os.listdir(_SOILGRIDS_PATH):
        if not fname.endswith(".yaml"):
            continue
        path = os.path.join(_SOILGRIDS_PATH, fname)
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        soil_data[fname] = data
        soil_regions[fname] = region_for_file(fname, region_lut)

    print(f"Loaded {len(soil_data)} soil files.")

    # --------------------------------------------------------------------------------------
    # 3) Extract features
    # --------------------------------------------------------------------------------------
    X = []
    soil_ids = []
    names_list = None

    for fname, data in soil_data.items():
        feats, vec, names = extract_soil_features(data)
        X.append(vec)
        soil_ids.append(fname)
        if names_list is None:
            names_list = names

    X = np.vstack(X)
    print("Matrix shape:", X.shape)  # (n_soils, n_features)

    # --------------------------------------------------------------------------------------
    # 4) PCA (fit on all loaded soils here; for RL use, fit offline on train-only)
    # --------------------------------------------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=5)  # keep enough components to explain 95% variance
    Z = pca.fit_transform(X_scaled)

    print(f"PCA reduced from {X.shape[1]} → {Z.shape[1]} dimensions.")
    print("Explained variance ratio:", pca.explained_variance_ratio_.sum())

    # --------------------------------------------------------------------------------------
    # 5) Save latent with regions
    # --------------------------------------------------------------------------------------
    regions = [soil_regions[sid] for sid in soil_ids]
    df_latent = pd.DataFrame(Z, index=soil_ids, columns=[f"PC{i + 1}" for i in range(Z.shape[1])])
    df_latent.insert(0, "region", regions)
    df_latent.to_csv("soil_latent_with_region.csv")
    print(df_latent.head())

    # --------------------------------------------------------------------------------------
    # 6) Plot PC1 vs PC2 colored by region
    # --------------------------------------------------------------------------------------
    pcx, pcy = 0, 1  # which PCs to plot
    unique_regions = sorted(set(regions))
    unique_regions.remove('unknown')

    # choose a color per region
    cmap = plt.get_cmap("tab10")
    color_for_region = {r: cmap(i % 10) for i, r in enumerate(unique_regions)}

    # group indices by region (for clean legend without duplicates)
    idx_by_region = defaultdict(list)
    for i, r in enumerate(regions):
        idx_by_region[r].append(i)

    plt.figure(figsize=(8, 6))
    for r in unique_regions:
        idxs = idx_by_region[r]
        plt.scatter(Z[idxs, pcx], Z[idxs, pcy], label=r, s=25, alpha=0.85,
                    c=[color_for_region[r]])

    # Optional: annotate points (can be busy if many)
    # for i, sid in enumerate(soil_ids):
    #     plt.text(Z[i, pcx], Z[i, pcy], sid.replace(".yaml",""), fontsize=6, alpha=0.7)

    plt.xlabel(f"PC{pcx + 1}")
    plt.ylabel(f"PC{pcy + 1}")
    plt.title("Soil PCA by region")
    plt.legend(title="Region", fontsize=9)
    plt.tight_layout()
    plt.show()


def extract_soil_features_test():
    with open(os.path.join(_SOILGRIDS_PATH, "soil_7.0966_53.247.yaml"), "r") as f:
        soil_dict = yaml.safe_load(f)
    # For demonstration you could paste your dict here:
    # soil_dict = {...}
    feats, vec, names = extract_soil_features(soil_dict)
    print(len(vec), "features")
    for k, v in list(feats.items())[:25]:
        print(k, "=", v)

def build_pca(use_region=False):
    def fname_to_coords(fname):
        # soil_{lon}_{lat}.yaml
        lon, lat = fname.replace("soil_", "").replace(".yaml", "").split("_")
        return float(lon), float(lat)

    def region_for_file(fname):
        lon, lat = fname_to_coords(fname)
        for region, coords in scenario_dict.items():
            for c in coords:
                if c[0] == lon and c[1] == lat:
                    return region
        return "unknown"

    # --- load region definitions
    if use_region:
        with open(os.path.join(_SCENARIO_PATH, "scenario_coords.yaml"), "r") as f:
            scenario_dict = yaml.safe_load(f)  # {'gelderland': [[lon, lat], ...], ...}

        # --- gather soils per region
        region_to_soils = {}
        for fn in os.listdir(_SOILGRIDS_PATH):
            if not fn.endswith(".yaml"):
                continue
            region = region_for_file(fn)
            region_to_soils.setdefault(region, []).append(fn)

        print({r: len(v) for r, v in region_to_soils.items()})

        # --- fit PCA per region
        for region, files in region_to_soils.items():
            X, soil_ids = [], []
            for fn in files:
                data = yaml.safe_load(open(os.path.join(_SOILGRIDS_PATH, fn)))
                _, vec, _ = extract_soil_features(data)
                X.append(vec)
                soil_ids.append(fn)

            if len(X) < 2:
                print(f"Skipping region {region}: not enough soils ({len(X)})")
                continue

            X = np.vstack(X)
            scaler = StandardScaler().fit(X)
            Xz = scaler.transform(X)
            pca = PCA(n_components=5).fit(Xz)

            joblib.dump(scaler, f"soil_scaler_{region}.joblib")
            joblib.dump(pca, f"soil_pca_{region}.joblib")
            print(f"{region}: {X.shape[1]} → {pca.n_components_} PCs (explained {pca.explained_variance_ratio_.sum():.2f})")
    else:
        # ---- Global PCA across ALL soils in the folder ----
        X, soil_ids, coords = [], [], []

        # Reuse fname_to_coords defined above
        for fn in os.listdir(_SOILGRIDS_PATH):
            if not fn.endswith(".yaml"):
                continue
            path = os.path.join(_SOILGRIDS_PATH, fn)
            with open(path, "r") as f:
                data = yaml.safe_load(f)

            _, vec, _ = extract_soil_features(data)
            X.append(vec)
            soil_ids.append(fn)
            lon, lat = fname_to_coords(fn)
            coords.append((lon, lat))

        if len(X) < 2:
            raise RuntimeError(f"Not enough soils in {_SOILGRIDS_PATH} to fit a global PCA.")

        X = np.vstack(X)
        print("Global matrix shape:", X.shape)

        # Fit scaler + PCA on ALL soils
        scaler = StandardScaler().fit(X)
        Xz = scaler.transform(X)
        pca = PCA(n_components=5).fit(Xz)  # or PCA(n_components=0.95) if you prefer variance target

        # Save models
        joblib.dump(scaler, "../configs/scenarios/soil_scaler_all.joblib")
        joblib.dump(pca, "../configs/scenarios/soil_pca_all.joblib")
        print(f"ALL: {X.shape[1]} → {pca.n_components_} PCs (explained {pca.explained_variance_ratio_.sum():.2f})")

        # Transform all soils to PCs and save a CSV with coords
        Z = pca.transform(Xz)
        df = pd.DataFrame(Z, columns=[f"PC{i + 1}" for i in range(Z.shape[1])])
        df.insert(0, "lat", [lat for (lon, lat) in coords])
        df.insert(0, "lon", [lon for (lon, lat) in coords])
        df.insert(0, "file", soil_ids)
        df.to_csv("soil_latent_all.csv", index=False)
        print(df.head())

# ---------------------- example usage ----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default="build", help="pca or build")
    args = parser.parse_args()

    if args.mode == "pca":
        do_pca()
    elif args.mode == "build":
        build_pca(use_region=False)
    else:
        extract_soil_features_test()
