"""Tests for nanotabicl/: checkpoint conversion, crop tables, loss, a training step."""

import datetime

import numpy as np
import pandas as pd
import pytest
import torch

from nanotabicl import checkpoint
from nanotabicl.crop_tables import CropTableConfig, CropTables
from nanotabicl.model import NanoTabICLv2
from nanotabicl.train import checkpoint_activations, evaluate_walk_forward, forward_quantiles, pinball_loss, train_step


SOS, EOS = 100, 170  # small crop calendar: window [SOS - 60, EOS] = 131 days = 5 blocks of 30 days
DAILY = ["tmin", "tmax", "tavg", "prec", "rad", "et0", "cwb", "fpar", "ssm", "rsm"]


def calendar_table(regions, seed=0, years=range(2005, 2021), jitters=3):
    """A small processed table in process_wofost_dataset.py's (calendar) layout."""
    rng = np.random.default_rng(seed)
    n_days = EOS - (SOS - 60) + 1
    rows = []
    for loc, country in enumerate(regions):
        lat, lon, awc = rng.uniform(47, 54), rng.uniform(0, 10), rng.uniform(15, 30)
        for year in years:
            rain = rng.uniform(0, 6, n_days)
            start = datetime.date(year, 1, 1) + datetime.timedelta(days=SOS - 1 - 60)
            for jitter in range(jitters):
                tavg = 12 + 8 * np.sin(np.arange(n_days) / 40) + rng.normal(0, 2, n_days)
                row = {"location_index": loc, "location_id": f"{lon}_{lat}", "country": country, "crop": "maize",
                       "year": year, "sowing_year": year, "jitter_index": jitter, "latitude": lat, "longitude": lon,
                       "awc": awc, "bulk_density": 1.3, "sos_doy": SOS, "eos_doy": EOS, "reached_maturity": True,
                       "window_start": start.isoformat(),
                       "yield_t_per_ha": 5 + 0.2 * awc + rain.mean() + rng.normal(0, 0.2)}
                daily = {"tavg": tavg, "tmin": tavg - 5, "tmax": tavg + 5, "prec": rain, "rad": rng.uniform(5e6, 2e7, n_days),
                         "et0": rng.uniform(1, 4, n_days), "fpar": np.clip(np.arange(n_days) - 60, 0, 90) * 1.0,
                         "ssm": rng.uniform(20, 40, n_days), "rsm": rng.uniform(250, 400, n_days)}
                daily["cwb"] = daily["prec"] - daily["et0"]
                for name in DAILY:
                    row.update({f"{name}_d{d:03d}": v for d, v in enumerate(daily[name])})
                rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def table_path(tmp_path_factory):
    regions = ["France"] * 12 + ["Germany"] * 11 + ["Belgium"] * 4  # Belgium: < 10 points, dropped
    path = tmp_path_factory.mktemp("table") / "wofost_maize_daily.parquet"
    calendar_table(regions).to_parquet(path, index=False)
    return str(path)


@pytest.fixture(scope="module")
def tables(table_path):
    return CropTables(table_path, CropTableConfig(min_context_years=3, max_context_years=6))


@pytest.fixture(scope="module")
def random_tables(table_path):
    return CropTables(table_path, CropTableConfig(episodes="random"))


def tiny_model():
    torch.manual_seed(0)
    return NanoTabICLv2(max_classes=0, out_dim=19, embed_dim=16, col_num_blocks=1, row_num_blocks=1, icl_num_blocks=1,
                        col_nhead=2, row_nhead=2, icl_nhead=2, n_cls_rows=8)


def test_official_checkpoint_conversion_is_exact():
    """Converted official TabICL weights give the official model's outputs (float64).

    The real checkpoint's settings (feature grouping "same", group size 3) and *all* weights randomised:
    a fresh TabICL has zero row-attention output projections, with which x doesn't reach the output at
    all and a wrong feature path would go unnoticed."""
    pytest.importorskip("tabicl")
    from tabicl._model.tabicl import TabICL

    torch.manual_seed(0)
    config = dict(max_classes=0, num_quantiles=9, embed_dim=32, col_num_blocks=2, row_num_blocks=2, icl_num_blocks=2,
                  col_nhead=4, row_nhead=4, icl_nhead=4, row_rope_interleaved=False, bias_free_ln=True,
                  col_feature_group="same", col_feature_group_size=3)
    official = TabICL(**config).double().eval()
    with torch.no_grad():
        for name, param in official.named_parameters():
            if not name.endswith("rope.freqs"):  # a parameter in tabicl, fixed (standard values) in practice
                param.add_(0.2 * torch.randn_like(param))
    nano = checkpoint.NanoTabICL(**checkpoint.nano_config_from_official(config)).double().eval()
    missing, unexpected = nano.load_state_dict(checkpoint.convert_official_state_dict(official.state_dict()), strict=False)
    assert not unexpected and all(k.endswith(".bias") for k in missing)

    for n_rows, n_train, n_cols in [(30, 24, 6), (60, 40, 11)]:
        x = torch.randn(2, n_rows, n_cols, dtype=torch.float64)
        x = (x - x[:, :n_train].mean(1, keepdim=True)) / x[:, :n_train].std(1, unbiased=False, keepdim=True)
        y = torch.randn(2, n_train, dtype=torch.float64)
        with torch.no_grad():
            expected = official._train_forward(x, y)
            assert torch.allclose(expected, nano(x, y), atol=1e-6)
            nano.official_feature_grouping = False  # the vendored grouping is one column off
            assert not torch.allclose(expected, nano(x, y), atol=1e-3)
            nano.official_feature_grouping = True


def test_features_are_cybench_columns_and_finite(tables):
    assert np.isfinite(tables.features).all()
    names = tables.feature_names
    assert names[:2] == ["cum_cwb_mean_0", "cum_cwb_mean_1"] and names[-9:] == [
        "sos_sin", "sos_cos", "eos_sin", "eos_cos", "loc_x", "loc_y", "loc_z", "awc", "bulk_density"]
    assert len(names) == (12 + 2 + 2) * 5 + 9  # 16 aggregated variables x 5 blocks + statics
    # standardised with the train years' mean/std per variable (pooled over blocks)
    train = tables.df["sowing_year"].between(2005, 2016).to_numpy()
    prec = tables.features[train][:, [n.startswith("prec_sum_") for n in names]]
    assert prec.mean() == pytest.approx(0, abs=1e-4) and prec.std(ddof=1) == pytest.approx(1, rel=1e-3)


def test_small_regions_are_dropped(tables):
    assert set(tables.country_points) == {"France", "Germany"} and tables.dropped_regions == {"Belgium": 4}


def _rows(tables, x):
    """Row indices of a sampled table (features are unique per row in the fixture)."""
    return np.array([int(np.flatnonzero((tables.features == r.numpy()).all(1))[0]) for r in x])


def test_structured_training_tables(tables):
    rng = np.random.default_rng(0)
    sizes = set()
    for _ in range(30):
        b = tables.sample_batch(rng, batch_size=2)
        n_rows, n_train = b["x"].shape[1], b["n_train"]
        n_points = n_rows - n_train
        assert n_train % n_points == 0  # n_points x n_context_years context rows
        sizes.add(n_points)
        for x in b["x"]:
            rows = _rows(tables, x)
            years, points = tables.year[rows], tables.point[rows]
            T = years[n_train]
            assert (years[:n_train] < T).all() and (years[n_train:] == T).all() and T <= 2016
            assert len(set(tables.df.loc[rows, "country"])) == 1  # one region per table
            assert len(set(zip(points, years))) == len(rows)  # one jitter per (point, year) cell
    assert min(sizes) >= 10 and max(sizes) > 10  # from 10 up to a region's size (France 12, Germany 11)


def test_random_training_tables(random_tables):
    rng = np.random.default_rng(0)
    for _ in range(10):
        b = random_tables.sample_batch(rng, batch_size=2)
        n_rows, n_train = b["x"].shape[1], b["n_train"]
        assert n_rows % 12 == 0 and 0.5 * n_rows <= n_train <= 0.9 * n_rows  # n_points x 12 train years
        rows = _rows(random_tables, b["x"][0])
        years = random_tables.year[rows]
        assert years.max() <= 2016 and len(set(years[:n_train])) > 1 and len(set(years[n_train:])) >= 1
        assert len(set(zip(random_tables.point[rows], years))) == n_rows


def test_training_tables_respect_the_year_split(tables):
    rng = np.random.default_rng(1)
    for _ in range(20):
        # recover the sampled rows via the sampler's own path
        n_points, n_years = 3, 4
        idx, n_train = tables._structured(rng, "France", n_points, n_years)
        years = tables.year[idx]
        T = years[n_train]
        assert (years[:n_train] < T).all() and (years[n_train:] == T).all()
        # one jitter per (point, year) cell
        cells = list(zip(tables.point[idx], tables.year[idx]))
        assert len(cells) == len(set(cells))


def test_walk_forward_tables(tables):
    for t in tables.walk_forward_tables([2019]):
        rows, n = t["rows"], t["n_train"]
        assert (tables.year[rows[:n]] < 2019).all() and (tables.year[rows[n:]] == 2019).all()
        assert set(tables.point[rows[n:]]) <= set(tables.point[rows[:n]])


def test_pinball_loss_is_minimized_by_true_quantiles():
    target = torch.randn(1, 2000)
    q = 9
    alphas = torch.linspace(0, 1, q + 2)[1:-1]
    true_q = torch.distributions.Normal(0, 1).icdf(alphas).expand(1, 2000, q)
    assert pinball_loss(true_q, target) < pinball_loss(true_q + 0.5, target)


def test_training_step_and_evaluation(tables):
    model = tiny_model()
    batch = tables.sample_batch(np.random.default_rng(0), 2)
    quantiles, y_query, _, _ = forward_quantiles(model, batch["x"], batch["y"], batch["n_train"], "cpu")
    loss = pinball_loss(quantiles, y_query)
    loss.backward()
    assert torch.isfinite(loss)
    metrics, preds = evaluate_walk_forward(model, tables, [2019], "cpu")
    assert len(preds) == 23 and np.isfinite(metrics["rmse"])  # France + Germany points at 2019


def test_matched_prior_tables_have_the_crop_batches_shapes(tables):
    from torch.utils.data import DataLoader

    from nanotabicl.prior_tables import MatchedPriorTables, crop_batch_shapes
    from nanotabicl.train import CropStream

    shapes = crop_batch_shapes(tables, batch_size=2, n_batches=3, seed=5)
    crop = iter(CropStream(tables, batch_size=2, seed=5))  # what `--data crop` trains on, step by step
    assert shapes == [(b["x"].shape[1], b["x"].shape[2], b["n_train"]) for b in (next(crop) for _ in shapes)]
    batches = list(DataLoader(MatchedPriorTables(shapes, batch_size=2, seed=0), batch_size=None, num_workers=2))
    assert [(b["x"].shape[1], b["x"].shape[2], b["n_train"]) for b in batches] == shapes  # in order across workers
    for b in batches:
        assert b["x"].shape[0] == b["y"].shape[0] == 2 and b["y"].shape[1] == b["x"].shape[1]
        assert torch.isfinite(b["x"]).all() and torch.isfinite(b["y"]).all()
        assert torch.allclose(b["x"].mean(dim=1), torch.zeros(1), atol=1e-4)  # standardised per table


def _grads_after_step(model, batch, micro_batch_size):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)  # lr 0: keep the weights, only compute gradients
    loss = train_step(model, optimizer, batch["x"], batch["y"], batch["n_train"], "cpu", micro_batch_size, grad_clip=1e9)
    return loss, {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}


def test_micro_batches_and_activation_checkpointing_give_the_same_update(tables):
    batch = tables.sample_batch(np.random.default_rng(0), 4)
    model = tiny_model().double()
    batch = {**batch, "x": batch["x"].double(), "y": batch["y"].double()}
    loss_full, grads_full = _grads_after_step(model, batch, micro_batch_size=4)
    loss_micro, grads_micro = _grads_after_step(model, batch, micro_batch_size=1)
    checkpoint_activations(model)
    loss_ckpt, grads_ckpt = _grads_after_step(model, batch, micro_batch_size=3)  # uneven split: 3 + 1 tables
    assert loss_micro == pytest.approx(loss_full) and loss_ckpt == pytest.approx(loss_full)
    for name, g in grads_full.items():
        assert torch.allclose(grads_micro[name], g, atol=1e-10) and torch.allclose(grads_ckpt[name], g, atol=1e-10), name


def test_activation_checkpointing_stores_less():
    """Bytes kept for the backward pass, with and without checkpointing."""
    x, y = torch.randn(1, 200, 40), torch.randn(1, 160)

    def saved_bytes(model):
        total = 0

        def pack(t):
            nonlocal total
            total += t.numel() * t.element_size()
            return t

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            model(x, y)
        return total

    model = tiny_model()
    plain = saved_bytes(model)
    checkpoint_activations(model)
    assert saved_bytes(model) < plain / 3


def test_regressor_matches_training_predictions_and_query_batching(tmp_path):
    from nanotabicl.regressor import NanoTabICLRegressor

    model = tiny_model()
    config = dict(max_classes=0, out_dim=19, embed_dim=16, col_num_blocks=1, row_num_blocks=1, icl_num_blocks=1,
                  col_nhead=2, row_nhead=2, icl_nhead=2, n_cls_rows=8)
    path = tmp_path / "tiny.pt"
    torch.save({"model_config": config, "state_dict": model.state_dict()}, path)
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(60, 5)).astype(np.float32), rng.normal(size=60).astype(np.float32)
    x[3, 2] = np.nan  # NaN -> 0, as in training
    reg = NanoTabICLRegressor(checkpoint=str(path), device="cpu").fit(x[:40], y[:40])
    pred = reg.predict(x[40:])
    batched = NanoTabICLRegressor(checkpoint=str(path), device="cpu", query_batch_size=3).fit(x[:40], y[:40]).predict(x[40:])
    np.testing.assert_allclose(pred, batched, rtol=1e-4, atol=1e-5)  # queries only attend to the context
    xt = torch.from_numpy(np.nan_to_num(x))[None]
    quantiles, _, mean, std = forward_quantiles(model.eval(), xt, torch.from_numpy(y)[None], 40, "cpu")
    np.testing.assert_allclose(pred, (quantiles.mean(-1) * std + mean)[0].detach().numpy(), rtol=1e-4, atol=1e-5)


def test_ensemble_views_match_tabicl():
    """Our port of TabICL's test-time ensemble builds the same views as tabicl's EnsembleGenerator."""
    pytest.importorskip("tabicl")
    from tabicl._sklearn.preprocessing import EnsembleGenerator

    from nanotabicl.ensemble import Ensemble

    rng = np.random.default_rng(0)
    x = np.column_stack([rng.normal(size=80), rng.exponential(size=80), rng.normal(size=80) * 100,
                         np.ones(80), rng.integers(0, 3, 80), rng.normal(size=80)])  # incl. a constant column
    x[5, 0] = 40.0  # an outlier
    context, query = x[:60], x[60:]
    ours = Ensemble(n_estimators=8, random_state=3).fit(context).transform(query)
    gen = EnsembleGenerator(classification=False, n_estimators=8, norm_methods=["none", "power"],
                            feat_shuffle_method="latin", outlier_threshold=4.0, random_state=3)
    gen.fit(context, np.zeros(60))
    theirs = [(xs[i][:60], xs[i][60:]) for xs, _ in gen.transform(query, mode="both").values() for i in range(len(xs))]
    assert len(ours) == len(theirs) == 8
    for (c1, q1), (c2, q2) in zip(ours, theirs):
        np.testing.assert_allclose(c1, c2, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(q1, q2, rtol=1e-6, atol=1e-8)


def _official_weights_cached() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    return isinstance(try_to_load_from_cache(checkpoint.HF_REPO, checkpoint.OFFICIAL_REGRESSOR), str)


@pytest.mark.slow
@pytest.mark.skipif(not _official_weights_cached(), reason="official TabICLv2 weights not downloaded")
def test_regressor_ensemble_matches_tabicl_regressor():
    """NanoTabICLRegressor(n_estimators=8) with the official weights = TabICLRegressor(n_estimators=8)."""
    pytest.importorskip("tabicl")
    from tabicl import TabICLRegressor

    from nanotabicl.regressor import NanoTabICLRegressor

    rng = np.random.default_rng(0)
    x = rng.normal(size=(120, 6)).astype(np.float32)
    x[:, 1] = np.exp(x[:, 1])  # skewed, so the power transform matters
    y = (2 * x[:, 0] + np.log(x[:, 1]) + rng.normal(0, 0.1, 120)).astype(np.float32)
    ours = NanoTabICLRegressor(checkpoint="official", device="cpu", n_estimators=8, random_state=0)
    pred = ours.fit(x[:100], y[:100]).predict(x[100:])
    official = TabICLRegressor(n_estimators=8, random_state=0, device="cpu", use_amp=False, use_fa3=False)
    raw = official.fit(x[:100], y[:100]).predict(x[100:], output_type="raw_quantiles")
    np.testing.assert_allclose(pred, raw.mean(-1), rtol=1e-3, atol=1e-3)
