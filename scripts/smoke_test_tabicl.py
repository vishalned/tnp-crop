"""Smoke test for the TabICL branch: build the tabular feature table (real
store or synthetic), fine-tune for a couple of epochs on CPU, predict on a
held-out split, and check the result is finite and beats a trivial
mean-predictor baseline.

    uv run python scripts/smoke_test_tabicl.py --store-dir data/processed/tnp_store_wheat
    uv run python scripts/smoke_test_tabicl.py --synthetic   # no real data needed

Not a real training run: `--epochs`/`--max-data-size` default small so it
finishes in a couple of minutes on CPU. The first run downloads the
~110 MB pretrained checkpoint from Hugging Face Hub (needs internet once;
cached under `~/.cache/huggingface` after that).
"""

import argparse
import os
import sys
import tempfile
import time

import numpy as np
import rootutils

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.components.crop_episode_dataset import CropStore, EpisodeConfig  # noqa: E402
from src.data.components.synthetic_crop_store import make_synthetic_store  # noqa: E402
from src.data.components.tabicl_table import TabICLTableConfig, build_table, split_table, table_to_xy  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Smoke-test the TabICL feature table + fine-tuning loop.")
    parser.add_argument("--store-dir", type=str, default=None, help="Training store (default: data/processed/tnp_store_wheat).")
    parser.add_argument("--synthetic", action="store_true", help="Use a small synthetic store instead of real data.")
    parser.add_argument("--profile", type=str, default="weekly", choices=["weekly", "dekadal"])
    parser.add_argument("--target", type=str, default="yield_t_per_ha")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-data-size", type=int, default=200, help="Rows per fine-tuning chunk (tabicl default: 10000).")
    args = parser.parse_args()

    tmp = tempfile.mkdtemp()
    if args.synthetic:
        store_dir = make_synthetic_store(os.path.join(tmp, "store"))
    else:
        store_dir = args.store_dir or os.path.join(root, "data", "processed", "tnp_store_wheat")
    print(f"store: {store_dir}")

    store = CropStore(store_dir, EpisodeConfig(countries=("France", "Germany", "Belgium", "United Kingdom", "Denmark", "Netherlands")))
    table = build_table(store, TabICLTableConfig(profile=args.profile))
    print(f"table: {len(table)} rows, {len(table.columns)} columns")

    splits = split_table(table)
    for name, df in splits.items():
        print(f"  {name}: {len(df)} rows")
    if min(len(df) for df in splits.values()) < 10:
        print("SMOKE TEST FAILED (too few rows in a split; use --synthetic or a bigger store)")
        sys.exit(1)

    X_train, y_train = table_to_xy(splits["train"], args.target)
    X_val, y_val = table_to_xy(splits["val"], args.target)
    X_test, y_test = table_to_xy(splits["test"], args.target)
    print(f"X_train {X_train.shape}, y_train {y_train.shape}")

    from tabicl import FinetunedTabICLRegressor

    model = FinetunedTabICLRegressor(
        epochs=args.epochs,
        device="cpu",
        verbose=True,
        early_stopping=False,  # too few epochs for patience to matter here
        max_data_size=args.max_data_size,
        n_estimators_finetune=1,
        n_estimators_validation=1,
        n_estimators_inference=2,
    )

    t0 = time.time()
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val, output_dir=os.path.join(tmp, "ckpt"))
    print(f"fine-tuned {args.epochs} epoch(s) in {time.time() - t0:.1f}s")

    preds = model.predict(X_test)
    mse = float(np.mean((preds - y_test.to_numpy()) ** 2))
    baseline_mse = float(np.mean((y_train.mean() - y_test.to_numpy()) ** 2))
    print(f"test MSE {mse:.4f} (mean-predictor baseline {baseline_mse:.4f})")

    ok = np.isfinite(preds).all() and np.isfinite(mse)
    print("SMOKE TEST PASSED" if ok else "SMOKE TEST FAILED (non-finite predictions)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
