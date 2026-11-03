"""Every experiment config in configs/experiment/ is what the runs of that experiment actually used: the settings saved in the old runs' config.json
(results/training/, copied by analysis/collect_results.py) match the composed Hydra config. (micro_batch_size, activation checkpointing and the DataLoader
workers are left out: they do not change the update.)"""

import glob
import json
import os

import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS, RESULTS = os.path.join(ROOT, "configs"), os.path.join(ROOT, "results", "training")
EXPERIMENTS = {  # (run folder, run tag) -> experiment config
    ("exp_1", "wofost"): "exp_1", ("exp_1", "scm"): "exp_1_scm",
    ("exp_1_eu_grid", "wofost"): "exp_1_eu_grid", ("exp_1_eu_grid", "scm"): "exp_1_eu_grid_scm",
    ("exp_1_eu_grid_dense", "wofost"): "exp_1_eu_grid_dense", ("exp_1_eu_grid_dense_omfix", "wofost"): "exp_1_eu_grid_dense_omfix",
    ("exp_2a_loguniform_points", "wofost"): "exp_2a_loguniform_points", ("exp_2b_loguniform_points_scm", "mix"): "exp_2b_loguniform_points_scm",
    ("exp_3_irrigation_map", "wofost"): "exp_3_irrigation_map", ("exp_4_obs_noise", "wofost"): "exp_4_obs_noise",
    ("sanity_permuted_yields", "wofost"): "sanity_permuted_yields", ("sanity_regions_nw", "wofost"): "sanity_regions_nw",
    ("sanity_regions_ce", "wofost"): "sanity_regions_ce", ("sanity_regions_south", "wofost"): "sanity_regions_south",
    ("sanity_regions_frbedk", "wofost"): "sanity_regions_frbedk",
}
SOURCE = {"crop": (0.0, None, None), "prior-matched": (1.0, "matched", 0), "mix-matched": (0.5, "matched", 10000)}  # old --data -> source


def compose_experiment(name, seed):
    with initialize_config_dir(config_dir=CONFIGS, version_base="1.3"):
        return compose(config_name="train", overrides=[f"experiment={name}", f"seed={seed}"])


@pytest.mark.parametrize("folder,tag", list(EXPERIMENTS))
def test_experiment_config_matches_the_runs_that_were_made(folder, tag):
    paths = sorted(glob.glob(f"{RESULTS}/{folder}/{tag}_seed*/config.json"))
    assert paths, f"no old runs of {folder}/{tag} in results/training"
    for path in paths:
        old = json.load(open(path))
        cfg = compose_experiment(EXPERIMENTS[(folder, tag)], old["seed"])
        c = OmegaConf.to_container(cfg, resolve=True)
        assert c["name"] == f"{folder}/{tag}_seed{old['seed']}"
        assert c["model"]["init"] == old["init"] and c["data"]["table"] == old["table"]
        for key in ("target", "horizon", "train_years", "val_years", "countries", "min_points", "max_points", "max_rows", "min_context_years",
                    "max_context_years"):
            assert c["data"][key] == old.get(key), key  # runs from before the country filter have no "countries": all countries
        assert c["data"]["episodes"] == old["crop_episodes"]
        for key in ("steps", "save_steps", "batch_size", "lr", "weight_decay", "grad_clip", "eval_every", "log_every", "warmup_fraction"):
            assert c["recipe"][key] == old[key], key
        assert old["stop_at"] in (None, old["steps"]) and c["recipe"]["stop_at"] in (None, old["steps"])
        fraction, shapes, offset = SOURCE[old["data"]]
        assert c["source"]["prior_fraction"] == fraction
        if shapes:
            assert (c["source"]["prior_shapes"], c["source"]["shape_seed_offset"]) == (shapes, offset)
        sampler = instantiate(cfg.sampler)
        assert type(sampler).__name__ == {"uniform": "UniformPoints", "loguniform": "LogUniformPoints"}[old.get("points_sampling", "uniform")]
        assert sampler.min_points == old.get("min_table_points")
        assert set(c["transforms"]) == {n for n, flag in (("noise", old.get("obs_noise")), ("permute", old.get("permute_yields"))) if flag}


def test_every_experiment_file_composes_and_builds_its_components():
    for path in sorted(glob.glob(f"{CONFIGS}/experiment/*.yaml")):
        cfg = compose_experiment(os.path.basename(path)[:-5], 0)
        instantiate(cfg.sampler)
        for t in cfg.transforms.values():
            instantiate(t)
        assert cfg.name and cfg.data.table
