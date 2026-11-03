"""The Hydra training entry (nanotabicl.train): a short CPU run reproduces the loss curve of the pre-refactor argparse loop, and writes the files
the rest of the code reads (tests/golden/train_v0.json was recorded with tag archive/pre-cleanup: scratch `small` model, 4 steps, batch 2)."""

import csv
import json
import os

import pytest
from hydra import compose, initialize_config_dir

from nanotabicl import checkpoint
from nanotabicl.train import train
from tests.test_nanotabicl import calendar_table

CONFIGS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")
GOLDEN = json.load(open(os.path.join(os.path.dirname(__file__), "golden", "train_v0.json")))
# the settings of the golden run (the old command line: --init scratch --size small --steps 4 --batch-size 2 --lr 1e-3 --warmup 2 ...)
OLD_RUN = ["model=scratch_small", "sampler=loguniform", "recipe.steps=4", "recipe.batch_size=2", "recipe.micro_batch_size=1", "recipe.lr=1e-3",
           "recipe.warmup=2", "recipe.warmup_fraction=null", "recipe.weight_decay=0.01", "recipe.eval_every=2", "recipe.log_every=1",
           "recipe.checkpoint_activations=false", "data.min_context_years=3", "data.max_context_years=6", "source.num_workers=0",
           "seed=0", "device=cpu"]


@pytest.fixture(scope="module")
def table_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("table") / "wofost_maize_daily.parquet"
    calendar_table(["France"] * 12 + ["Germany"] * 11 + ["Belgium"] * 4).to_parquet(path, index=False)
    return str(path)


def run(overrides, out_dir, table_path):
    with initialize_config_dir(config_dir=CONFIGS, version_base="1.3"):
        cfg = compose(config_name="train", overrides=[f"out_dir={out_dir}", f"data.table={table_path}", *overrides])
    train(cfg)
    return list(csv.DictReader(open(os.path.join(out_dir, "log.csv"))))


def losses(rows):
    return [r["source"] for r in rows], [float(r["loss"]) for r in rows], [float(r["val_rmse"]) if r["val_rmse"] else None for r in rows]


def test_crop_run_reproduces_the_old_loss_curve_and_writes_its_files(tmp_path, table_path):
    rows = run([*OLD_RUN, "data.val_years=[]"], tmp_path, table_path)  # no validation: the training does not depend on it (fast)
    sources, loss, _ = losses(rows)
    assert sources == GOLDEN["plain"]["source"]
    assert loss == pytest.approx(GOLDEN["plain"]["loss"], rel=1e-7)
    for name in ("config.json", "log.csv", "last.pt"):
        assert os.path.exists(tmp_path / name), name
    saved = json.load(open(tmp_path / "config.json"))
    assert saved["data"]["table"] == table_path and saved["sampler"]["_target_"].endswith("LogUniformPoints")
    # the settings the evaluation reads come back from the checkpoint's saved config (new nested format; the old flat one is read too)
    import torch

    args = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=False)["args"]
    assert checkpoint.trained_setting(args, "horizon") == "eos" and checkpoint.trained_setting(args, "min_points") == 10
    assert checkpoint.trained_setting({"horizon": "eos", "data": "crop", "min_points": 7}, "min_points") == 7


@pytest.mark.slow
def test_crop_run_with_validation_reproduces_the_old_validation_rmse_and_keeps_best(tmp_path, table_path):
    rows = run(OLD_RUN, tmp_path, table_path)
    _, loss, val = losses(rows)
    assert loss == pytest.approx(GOLDEN["plain"]["loss"], rel=1e-7)
    assert [v for v in val if v] == pytest.approx([v for v in GOLDEN["plain"]["val_rmse"] if v], rel=1e-7)
    assert os.path.exists(tmp_path / "best.pt")


@pytest.mark.slow
def test_mix_matched_run_reproduces_the_old_loss_curve(tmp_path, table_path):
    rows = run([*OLD_RUN, "source=mix_matched", "source.num_workers=0"], tmp_path, table_path)
    sources, loss, _ = losses(rows)
    assert sources == GOLDEN["mix"]["source"]
    assert loss == pytest.approx(GOLDEN["mix"]["loss"], rel=1e-7)


def test_noise_and_permute_transforms_run_through_the_config(tmp_path, table_path):
    rows = run(["model=tiny", "recipe.steps=3", "recipe.batch_size=2", "recipe.eval_every=3", "recipe.log_every=1", "recipe.warmup=1",
                "recipe.warmup_fraction=null", "recipe.checkpoint_activations=false", "data.min_context_years=3", "data.max_context_years=6",
                "sampler=loguniform", "+transform@transforms.noise=obs_noise", "+transform@transforms.permute=permute_yields", "device=cpu"],
               tmp_path, table_path)
    assert len(rows) == 3 and os.path.exists(tmp_path / "noise_params.csv")
    assert len(open(tmp_path / "noise_params.csv").read().splitlines()) == 1 + 3 * 2  # header + 3 steps x 2 tables
    assert set(json.load(open(tmp_path / "config.json"))["transforms"]) == {"noise", "permute"}
