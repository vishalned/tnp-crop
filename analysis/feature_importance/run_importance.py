"""Grouped permutation importance of one model on the CY-Bench folds or the simulated tables.

    poetry run python /.../run_importance.py --model wofost --data cybench --countries DE FR ...
    poetry run python /.../run_importance.py --model scm --data sim

Per fold/table and group: ONE random row permutation of the stacked context+query rows is applied to all of
the group's columns (columns stay in place), then the model predicts again. Baseline (all features intact) is
predicted in the same run. 3 permutation seeds. Metrics on CY-Bench are pooled over a country's folds like the
benchmark (`compute_report_metrics`); on simulated tables per table. Partial results are written per unit
(model, data, country or shard) and skipped on a rerun.
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from analysis.feature_importance.groups import build_groups, permute_group

ROOT = "/projects/prjs1788/tnp-crop/feature_importance"
CKPT = {
    "default_tabicl": "official",
    "scm": "/projects/prjs1788/tnp-crop/nanotabicl/exp_1_eu_grid/scm_seed0/last.pt",
    "wofost": "/projects/prjs1788/tnp-crop/nanotabicl/exp_1_eu_grid_dense/wofost_seed0/last.pt",
}
# existing exp_1 benchmark runs, to check that our baseline reproduces them: (experiment, run suffix)
REFERENCE = {
    "default_tabicl": ("exp_1_eu_grid_dense", "tabicl_eos_tabicl_default_single"),
    "scm": ("exp_1_eu_grid", "tabicl_wofost_eos_scm_seed0"),
    "wofost": ("exp_1_eu_grid_dense", "tabicl_wofost_eos_wofost_seed0"),
}
COUNTRY_IDX = {c: i for i, c in enumerate(["DE", "FR", "IT", "ES", "RO", "HU", "PL"])}


def make_regressor(model: str, device: str):
    from nanotabicl.regressor import NanoTabICLRegressor

    # default TabICL = TabICLRegressor(n_estimators=1) as in the exp_1 reference; our checkpoints: one plain pass
    return NanoTabICLRegressor(checkpoint=CKPT[model], device=device, n_estimators=1 if model == "default_tabicl" else None)


def predict(reg, x_all: np.ndarray, y_ctx: np.ndarray, n_ctx: int) -> np.ndarray:
    reg.fit(x_all[:n_ctx], y_ctx)
    return reg.predict(x_all[n_ctx:])


def seed_for(perm_seed: int, *keys: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([perm_seed, *keys]))


def anomaly_and_nrmse(df: pd.DataFrame) -> tuple:
    from cybench.evaluation.aggregated_metrics import compute_report_metrics

    m = compute_report_metrics(df, "targets", "preds")
    return m["region_year"]["nrmse"], m["anomaly"]["r_pooled"]


def reference_check(model: str, country: str, folds, base_preds: dict) -> dict:
    exp, run = REFERENCE[model]
    d = f"/projects/prjs1788/tnp-crop/cybench_output/{exp}/maize_{country}_{run}/walk_forward"
    out = {}
    for y, p in base_preds.items():
        ref = pd.read_csv(f"{d}/{y}/42/test_preds.csv")
        mine = pd.DataFrame({"adm_id": folds[f"{y}_adm_id"], "mine": p})
        merged = ref.merge(mine, on="adm_id")
        out[y] = float(np.abs(merged["preds"] - merged["mine"]).max()) if len(merged) == len(ref) else float("nan")
    return out


def run_cybench(args, reg):
    for country in args.countries:
        part = f"{ROOT}/parts/{args.model}_cybench_{country}.csv"
        if os.path.exists(part):
            print("skip", part)
            continue
        t0 = time.time()
        folds = np.load(f"{ROOT}/folds/folds_{country}.npz", allow_pickle=True)  # our own dump (adm_id strings)
        names = list(folds["columns"])
        groups = build_groups(names)
        years = [int(y) for y in folds["fold_years"]]
        data = {y: (np.concatenate([folds[f"{y}_X_ctx"], folds[f"{y}_X_q"]]), folds[f"{y}_y_ctx"], len(folds[f"{y}_y_ctx"])) for y in years}
        meta = pd.concat([pd.DataFrame({"adm_id": folds[f"{y}_adm_id"], "year": y, "targets": folds[f"{y}_y_q"]}) for y in years], ignore_index=True)

        base = {y: predict(reg, *data[y]) for y in years}
        check = reference_check(args.model, country, folds, base)
        print(f"{args.model} {country}: baseline vs existing test_preds, max |diff| per year: {check}", flush=True)
        b_nrmse, b_r = anomaly_and_nrmse(meta.assign(preds=np.concatenate([base[y] for y in years])))
        rows = []
        for gi, ((group, view), cols) in enumerate(sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0]))):
            for ps in args.perm_seeds:
                preds = []
                for y in years:
                    x, yc, n = data[y]
                    preds.append(predict(reg, permute_group(x, cols, seed_for(ps, COUNTRY_IDX[country], y, gi)), yc, n))
                nrmse, r = anomaly_and_nrmse(meta.assign(preds=np.concatenate(preds)))
                rows.append({"model": args.model, "seed": 0, "data": "cybench", "unit": country, "country": country,
                             "group": group, "group_view": view, "n_columns": len(cols), "perm_seed": ps,
                             "baseline_nrmse": b_nrmse, "permuted_nrmse": nrmse, "delta_nrmse": nrmse - b_nrmse,
                             "baseline_anomaly_r": b_r, "permuted_anomaly_r": r, "delta_anomaly_r": r - b_r,
                             "n_rows": len(meta)})
        pd.DataFrame(rows).to_csv(part, index=False)
        json.dump({"max_abs_diff_per_year": check, "seconds": time.time() - t0}, open(part.replace(".csv", "_check.json"), "w"))
        print(f"{args.model} {country} done in {time.time() - t0:.0f}s", flush=True)


def run_sim(args, reg):
    from cybench.evaluation.aggregated_metrics import calc_nrmse

    index = json.load(open(f"{ROOT}/sim_tables/index.json"))
    for shard in sorted({t["shard"] for t in index}):
        part = f"{ROOT}/parts/{args.model}_sim_{shard}.csv"
        if os.path.exists(part):
            print("skip", part)
            continue
        t0 = time.time()
        z = np.load(f"{ROOT}/sim_tables/sim_tables_{shard}.npz")
        groups = sorted(build_groups(list(z["columns"])).items(), key=lambda kv: (kv[0][1], kv[0][0]))
        rows = []
        for t in [t for t in index if t["shard"] == shard]:
            tid = t["table_id"]
            x, y, n = z[f"{tid}_x"], z[f"{tid}_y"], int(z[f"{tid}_n_train"])
            y_q = y[n:]
            b = calc_nrmse(y_q, predict(reg, x, y[:n], n))
            tix = int(tid.rsplit("_", 1)[1])
            for gi, ((group, view), cols) in enumerate(groups):
                for ps in args.perm_seeds:
                    p = calc_nrmse(y_q, predict(reg, permute_group(x, cols, seed_for(ps, 10_000 + tix, gi)), y[:n], n))
                    rows.append({"model": args.model, "seed": 0, "data": "simulated", "unit": tid, "country": t["country"],
                                 "group": group, "group_view": view, "n_columns": len(cols), "perm_seed": ps,
                                 "baseline_nrmse": b, "permuted_nrmse": p, "delta_nrmse": p - b,
                                 "baseline_anomaly_r": np.nan, "permuted_anomaly_r": np.nan, "delta_anomaly_r": np.nan,
                                 "n_rows": len(y_q)})
        pd.DataFrame(rows).to_csv(part, index=False)
        print(f"{args.model} sim shard {shard} done in {time.time() - t0:.0f}s", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(CKPT))
    p.add_argument("--data", required=True, choices=["cybench", "sim"])
    p.add_argument("--countries", nargs="+", default=["HU", "PL", "RO", "ES", "IT", "FR", "DE"])  # small first: the baseline check shows early
    p.add_argument("--perm-seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    reg = make_regressor(args.model, args.device)
    run_cybench(args, reg) if args.data == "cybench" else run_sim(args, reg)


if __name__ == "__main__":
    main()
