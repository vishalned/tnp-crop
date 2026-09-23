"""Tests for the crop TNP data pipeline and model, on a synthetic store."""

import datetime

import numpy as np
import pytest
import torch

from src.data.components.crop_episode_dataset import (
    REFERENCE_DATE,
    EpisodeConfig,
    aggregate,
    collate_episodes,
    intervals_for_profile,
)
from src.data.components.crop_vocab import MODALITY_ID, WEATHER_VARIABLES
from src.data.components.synthetic_crop_store import make_synthetic_store
from src.data.crop_datamodule import CropEpisodeDataModule
from src.models.components.tnpd import TNPD, Batch

LABEL_IDS = {MODALITY_ID["yield"], MODALITY_ID["phenology_maturity"]}


@pytest.fixture(scope="module")
def datamodule(tmp_path_factory):
    store = make_synthetic_store(str(tmp_path_factory.mktemp("store")))
    dm = CropEpisodeDataModule(store_dir=store, batch_size=3, max_points=4, max_context_years=6, val_episodes=6, test_episodes=6)
    dm.setup("fit")
    return dm


def small_net(**kw) -> TNPD:
    torch.manual_seed(0)
    return TNPD(dim_y=1, d_model=32, emb_depth=2, dim_feedforward=64, nhead=4, dropout=0.0, num_layers=2, num_modalities=17, **kw)


def _season_year(t: torch.Tensor) -> np.ndarray:
    return np.array([(REFERENCE_DATE + datetime.timedelta(days=float(v))).year for v in t])


def test_aggregation_sums_fluxes_and_averages_states():
    daily = np.arange(14 * len(WEATHER_VARIABLES), dtype=np.float32).reshape(14, -1)
    out = aggregate(daily, np.array([[0, 7], [7, 7]]))
    for i, v in enumerate(WEATHER_VARIABLES):
        expected = daily[:7, i].sum() if v in ("precip", "radiation") else daily[:7, i].mean()
        assert out[0, i] == pytest.approx(expected)
    daily[3, 0] = np.nan
    assert np.isnan(aggregate(daily, np.array([[0, 7]]))[0, 0])  # a missing day drops the bucket


def test_irregular_intervals_have_gaps_and_fixed_length():
    cfg = EpisodeConfig(countries=["x"], window_days=300)
    iv = intervals_for_profile("irregular", cfg, np.random.default_rng(0))
    assert (iv[:, 1] == 7).all() and (iv[:, 0] + 7 <= 300).all()
    assert (np.diff(iv[:, 0]) >= 7).all() and (np.diff(iv[:, 0]) > 7).any()


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_episode_invariants(datamodule, split):
    ds = {"train": datamodule.data_train, "val": datamodule.data_val, "test": datamodule.data_test}[split]
    target_years = {"train": range(2005, 2017), "val": (2017, 2018), "test": (2019, 2020)}[split]
    episodes = [next(iter(ds)) for _ in range(5)] if split == "train" else [ds[i] for i in range(5)]
    for e in episodes:
        info = e["info"]
        T = info["target_year"]
        assert T in target_years
        # targets: one yield + one maturity query per point, all at year T
        assert len(e["xt"]) == 2 * info["num_points"]
        assert set(_season_year(e["xt"][:, 2])) == {T}
        # context labels: C years per point, all before T (train: inside the train pool)
        is_label = np.isin(e["mc"].numpy(), list(LABEL_IDS))
        years = _season_year(e["xc"][is_label, 2])
        assert (years < T).all()
        if split == "train":
            assert (years >= 2005).all()
        assert is_label.sum() == 2 * info["num_points"] * info["context_years"]
        assert torch.isfinite(e["yc"]).all() and torch.isfinite(e["yt"]).all()


def test_eval_episodes_are_deterministic(datamodule):
    a, b = datamodule.data_val[3], datamodule.data_val[3]
    assert torch.equal(a["xc"], b["xc"]) and torch.equal(a["yt"], b["yt"])


def test_two_stream_encoder_equals_masked_joint_pass():
    """The context/target streams compute exactly the original TNP masked pass."""
    net = small_net().eval()
    B, nc, nt = 2, 30, 5
    batch = Batch(xc=torch.rand(B, nc, 4) * 50, yc=torch.randn(B, nc, 1), mc=torch.randint(0, 17, (B, nc)),
                  xt=torch.rand(B, nt, 4) * 50, yt=torch.randn(B, nt, 1), mt=torch.randint(0, 17, (B, nt)))
    ours = net.encode(batch)

    tok_ctx, tok_tar = net.construct_input(batch)
    h = net.embedder(torch.cat([tok_ctx, tok_tar], dim=1))
    mask = torch.full((nc + nt, nc + nt), float("-inf"))
    mask[:, :nc] = 0.0  # original TNP: everyone attends to context only
    for layer in net.layers:
        h = torch.nn.TransformerEncoderLayer.forward(layer, h, src_mask=mask)
    assert torch.allclose(ours, h[:, nc:], atol=1e-5)


def test_padding_does_not_change_predictions_or_loss(datamodule):
    net = small_net().eval()
    episodes = [datamodule.data_val[i] for i in range(3)]
    batch = Batch.from_dict(collate_episodes(episodes))
    assert not batch.mask_c.all()  # the episodes differ in length, so there is padding
    joint = net(batch, reduce_ll=False)["tar_ll_tokens"]
    for b, e in enumerate(episodes):
        alone = net(Batch.from_dict(collate_episodes([e])), reduce_ll=False)["tar_ll_tokens"][0]
        assert torch.allclose(joint[b, : len(alone)], alone, atol=1e-4)

    # padded target positions contribute exactly zero, whatever their values
    loss = net(batch)["loss"]
    garbage = Batch(**{**batch.__dict__, "yt": torch.where(batch.mask_t[..., None], batch.yt, torch.full_like(batch.yt, 1e6))})
    assert torch.equal(net(garbage)["loss"], loss)


def test_forward_backward_on_a_batch(datamodule):
    net = small_net()
    batch = next(iter(datamodule.train_dataloader()))
    out = net(Batch.from_dict(batch))
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert net.mask_value.grad is not None  # the learnable mask value is used and trained


def test_countries_are_sampled_uniformly_not_by_point_count(datamodule):
    # the synthetic store has 30 French points but only 4 Belgian ones
    rng = np.random.default_rng(0)
    sampler = datamodule.data_train.sampler
    counts = {}
    for _ in range(600):
        country = sampler.sample(rng)[1].country
        counts[country] = counts.get(country, 0) + 1
    assert len(counts) == 6
    assert all(60 <= n <= 140 for n in counts.values()), counts  # ~100 each
