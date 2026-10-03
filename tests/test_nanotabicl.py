"""Tests for nanotabicl/: checkpoint conversion, crop tables, loss, a training step."""

import numpy as np
import pandas as pd
import pytest
import torch

from nanotabicl import checkpoint
from nanotabicl.crop_tables import CropTableConfig, CropTables
from nanotabicl.model import NanoTabICLv2
from nanotabicl.train import evaluate_walk_forward, forward_quantiles, pinball_loss


@pytest.fixture(scope="module")
def table_path(tmp_path_factory):
    """A small processed table in process_wofost_dataset.py's layout."""
    rng = np.random.default_rng(0)
    rows = []
    regions = ["France"] * 12 + ["Germany"] * 11 + ["Belgium"] * 4  # Belgium: < 10 points, dropped
    for loc, country in enumerate(regions):
        lat, lon, awc = rng.uniform(47, 54), rng.uniform(0, 10), rng.uniform(0.15, 0.3)
        for year in range(2005, 2021):
            rain = rng.uniform(1, 3, 70)
            for jitter in range(3):
                row = {"location_index": loc, "location_id": f"{lon}_{lat}", "country": country, "crop": "wheat",
                       "year": year + 1, "sowing_year": year, "jitter_index": jitter, "latitude": lat, "longitude": lon,
                       "awc": awc, "bulk_density": 1.3, "reached_maturity": True,
                       "yield_t_per_ha": 5 + 20 * awc + rain.mean() + rng.normal(0, 0.2)}
                row.update({f"prec_d{d:03d}": rain[d] for d in range(70)})
                row.update({f"tmax_d{d:03d}": 15 + rng.normal() for d in range(70)})
                row.update({f"ssm_d{d:03d}": (0.3 if d < 50 else np.nan) for d in range(70)})
                rows.append(row)
    path = tmp_path_factory.mktemp("table") / "wofost_wheat_daily.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
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
    """Converted official TabICL weights give the official model's outputs (float64)."""
    pytest.importorskip("tabicl")
    from tabicl._model.tabicl import TabICL

    torch.manual_seed(0)
    config = dict(max_classes=0, num_quantiles=9, embed_dim=32, col_num_blocks=2, row_num_blocks=2, icl_num_blocks=2,
                  col_nhead=4, row_nhead=4, icl_nhead=4, row_rope_interleaved=False, bias_free_ln=True)
    official = TabICL(**config).double().eval()
    with torch.no_grad():  # make the zero-initialized parts non-zero so they're tested too
        for name, param in official.named_parameters():
            if "query_mlp.2" in name:
                param.normal_(0, 0.1)
    nano = NanoTabICLv2(**checkpoint.nano_config_from_official(config)).double().eval()
    missing, unexpected = nano.load_state_dict(checkpoint.convert_official_state_dict(official.state_dict()), strict=False)
    assert not unexpected and all(k.endswith(".bias") for k in missing)

    x = torch.randn(2, 30, 6, dtype=torch.float64)
    x = (x - x[:, :24].mean(1, keepdim=True)) / x[:, :24].std(1, unbiased=False, keepdim=True)  # nano standardizes inside
    y = torch.randn(2, 24, dtype=torch.float64)
    with torch.no_grad():
        assert torch.allclose(official._train_forward(x, y), nano(x, y), atol=1e-10)


def test_features_are_aggregated_and_finite(tables):
    assert np.isfinite(tables.features).all()  # NaN soil moisture outside the season filled
    names = tables.feature_names
    assert names[:4] == ["latitude", "longitude", "awc", "bulk_density"]
    assert sum(n.startswith("prec_b") for n in names) == 70 // 7
    # precipitation is summed per bucket, temperature averaged
    i = names.index("prec_b00")
    assert tables.features[0, i] == pytest.approx(tables.df.loc[0, [f"prec_d{d:03d}" for d in range(7)]].sum(), rel=1e-5)


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
