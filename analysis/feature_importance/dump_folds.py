"""Dump the CY-Bench walk-forward folds (features, targets, region/year) of a country, exactly as the
benchmark's pipeline feeds the model, without touching the benchmark repo.

Run in the benchmark's poetry env (see slurm/dump_folds.sh):
    poetry run python /.../dump_folds.py --country DE --out /projects/prjs1788/tnp-crop/feature_importance/folds

It runs the benchmark's own `run_experiments.main` (walk_forward, same overrides and frozen screening
dir as exp_1_eu_grid_dense), with the model's estimator replaced by a recorder that stores
(context X, y, query X, query y, adm_id, year) per fold and predicts zeros, so no checkpoint is loaded.
"""

import argparse
import json
import sys

import numpy as np

SCREENING = ("/projects/prjs1788/tnp-crop/cybench_output/exp_1_eu_grid_dense/"
             "maize_{c}_tabicl_wofost_eos_wofost_seed0/screening")  # + the split folder (test years differ per country)
RECORDS = []


class Recorder:
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--country", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    from cybench.models import tabular_foundation_model as tfm

    tfm.TabICLWofostModel._make_estimator = lambda self, device=None: Recorder()

    def predict(self, dataset, **kw):
        X_df, y_df = dataset.xy
        idx = X_df.index.to_frame(index=False)
        RECORDS.append({"X_ctx": np.asarray(self._train_X, np.float32), "y_ctx": np.asarray(self._train_y, np.float32),
                        "X_q": X_df.to_numpy(np.float32), "y_q": y_df.values.ravel().astype(np.float32),
                        "adm_id": idx["adm_id"].astype(str).to_numpy(), "year": idx["year"].astype(int).to_numpy(),
                        "columns": list(X_df.columns)})
        return np.zeros(len(X_df)), {}

    tfm.TabularFoundationModel.predict = predict

    import glob
    import os

    c = args.country
    frozen = os.path.dirname(glob.glob(SCREENING.format(c=c) + "/*/optimal_model.yaml")[0])
    sys.argv = ["run_experiments", "dataset/crop=maize", f"dataset.country={c}", "dataset.framework=pandas",
                "dataset/temporal=feature_design", "dataset.temporal.season.end_of_sequence=eos",
                "dataset.exclude_features=[vpd,ndvi,drainage_class]", "model=tabicl_wofost", "model.device=cpu",
                "model.allow_cpu_fallback=true", "experiment.device=cpu", "experiment.n_jobs=1",
                "experiment.n_repetitions=1", "experiment.name=feature_importance_dump", "validation=walk_forward",
                f"validation.frozen_screening_dir={frozen}",
                f"hydra.run.dir={args.out}/hydra_tmp/{c}"]
    import runpy

    import cybench.runs

    # as a script (hydra's config_path is relative to the file); the patches above live in this process
    runpy.run_path(f"{cybench.runs.__path__[0]}/run_experiments.py", run_name="__main__")

    arrays = {"columns": np.array(RECORDS[0]["columns"])}
    years = []
    for r in RECORDS:
        assert r["columns"] == list(arrays["columns"])
        y = int(np.unique(r["year"])[0])
        assert len(np.unique(r["year"])) == 1, "a fold's queries span several years"
        years.append(y)
        for k in ("X_ctx", "y_ctx", "X_q", "y_q", "adm_id", "year"):
            arrays[f"{y}_{k}"] = r[k]
    arrays["fold_years"] = np.array(years)
    np.savez_compressed(f"{args.out}/folds_{c}.npz", **arrays)
    print(c, "folds", years, "features", len(arrays["columns"]), json.dumps({y: len(r["y_q"]) for y, r in zip(years, RECORDS)}))


if __name__ == "__main__":
    main()
