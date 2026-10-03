"""Checkpoints: the official pretrained TabICLv2 weights -> NanoTabICLv2, and
our own save/load.

The official checkpoints (`tabicl-regressor-v2-*.ckpt`,
`tabicl-classifier-v2-*.ckpt`) live on the Hugging Face Hub (repo
`jingang/TabICL`) and hold `{"config": ..., "state_dict": ...}` for the
`tabicl` package's `TabICL` module. NanoTabICLv2 is the same architecture
with different parameter names, so conversion is a key rename plus two
reshapes. Verified against `tabicl._model.tabicl.TabICL._train_forward`:
identical outputs in float64 (`tests/test_nanotabicl.py`).

The regression checkpoint uses LayerNorm without bias; NanoTabICLv2's
LayerNorms have a bias, which is set to zero (identical function) and then
trains like any other parameter.
"""

import os
import re
from typing import Optional

import torch

from nanotabicl.model import NanoTabICLv2

HF_REPO = "jingang/TabICL"
OFFICIAL_REGRESSOR = "tabicl-regressor-v2-20260212.ckpt"

# Official TabICL options that NanoTabICLv2 hardcodes; a checkpoint with other values can't be converted.
_REQUIRED = {
    "col_affine": False, "col_target_aware": True, "row_rope_interleaved": False, "row_rope_base": 100000,
    "ff_factor": 2, "activation": "gelu", "norm_first": True,
    "col_ssmax": ("qassmax-mlp-elementwise", True), "icl_ssmax": ("qassmax-mlp-elementwise", True),
    "col_feature_group": ("same", True),
}


def download_official(filename: str = OFFICIAL_REGRESSOR, cache_dir: Optional[str] = None) -> str:
    """Path to an official TabICLv2 checkpoint, downloaded from the Hugging Face Hub if not cached."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=HF_REPO, filename=filename, cache_dir=cache_dir)


def nano_config_from_official(config: dict) -> dict:
    """NanoTabICLv2 constructor kwargs for an official TabICL config."""
    for key, allowed in _REQUIRED.items():
        allowed = allowed if isinstance(allowed, tuple) else (allowed,)
        if key in config and config[key] not in allowed:
            raise ValueError(f"Official config has {key}={config[key]!r}; NanoTabICLv2 only supports {allowed}.")
    regression = config.get("max_classes", 10) == 0
    return dict(
        max_classes=config.get("max_classes", 10),
        out_dim=config.get("num_quantiles", 999) if regression else config.get("max_classes", 10),
        embed_dim=config.get("embed_dim", 128),
        col_num_blocks=config.get("col_num_blocks", 3),
        row_num_blocks=config.get("row_num_blocks", 3),
        icl_num_blocks=config.get("icl_num_blocks", 12),
        col_nhead=config.get("col_nhead", 8),
        row_nhead=config.get("row_nhead", 8),
        icl_nhead=config.get("icl_nhead", 8),
        feature_group_size=config.get("col_feature_group_size", 3),
        n_cls_cols=config.get("row_num_cls", 4),
        n_cls_rows=config.get("col_num_inds", 128),
    )


_RENAMES = [
    (r"^col_embedder\.in_linear\.", "x_embed."),
    (r"^col_embedder\.y_encoder\.", "y_embed_in."),
    (r"^icl_predictor\.y_encoder\.", "y_embed_icl."),
    (r"^row_interactor\.cls_tokens$", "row_cls_tokens"),
    (r"^row_interactor\.out_ln\.", "row_ln."),
    (r"^icl_predictor\.ln\.", "out_ln."),
    (r"^icl_predictor\.decoder\.", "out_mlp."),
    (r"^col_embedder\.tf_col\.blocks\.(\d+)\.ind_vectors$", r"col_blocks.\1.inducing_vectors"),
    (r"^col_embedder\.tf_col\.blocks\.(\d+)\.multihead_attn1\.", r"col_blocks.\1.tfm1."),
    (r"^col_embedder\.tf_col\.blocks\.(\d+)\.multihead_attn2\.", r"col_blocks.\1.tfm2."),
    (r"^row_interactor\.tf_row\.blocks\.(\d+)\.", r"row_blocks.\1."),
    (r"^icl_predictor\.tf_icl\.blocks\.(\d+)\.", r"icl_blocks.\1."),
    (r"\.attn\.", "."),
    (r"\.linear1\.", ".mlp.0."),
    (r"\.linear2\.", ".mlp.2."),
    (r"\.norm1\.", ".ln_attn."),
    (r"\.norm2\.", ".ln_mlp."),
]


def convert_official_state_dict(state_dict: dict) -> dict:
    out = {}
    for key, value in state_dict.items():
        if key.endswith("rope.freqs"):  # NanoTabICLv2 recomputes the (identical) RoPE frequencies
            continue
        name = key
        for pattern, repl in _RENAMES:
            name = re.sub(pattern, repl, name)
        if name == "row_cls_tokens":
            value = value[None, None]  # (n_cls, d) -> (1, 1, n_cls, d)
        elif name.endswith("inducing_vectors"):
            value = value[None]  # (n_ind, d) -> (1, n_ind, d)
        out[name] = value
    return out


def load_official(path: str) -> NanoTabICLv2:
    """NanoTabICLv2 initialized from an official TabICLv2 checkpoint file."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = NanoTabICLv2(**nano_config_from_official(ckpt["config"]))
    missing, unexpected = model.load_state_dict(convert_official_state_dict(ckpt["state_dict"]), strict=False)
    # only the LayerNorm biases may be absent (bias-free LN in the regression checkpoint; they init to zero)
    bad_missing = [k for k in missing if not (k.endswith(".bias") and ("ln" in k.split(".")[-2]))]
    if bad_missing or unexpected:
        raise ValueError(f"Checkpoint doesn't match NanoTabICLv2: missing {bad_missing}, unexpected {unexpected}.")
    return model


def save(path: str, model: NanoTabICLv2, model_config: dict, **extra) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"model_config": model_config, "state_dict": model.state_dict(), **extra}, path)


def load(path: str) -> tuple:
    """(model, checkpoint dict) from a checkpoint written by `save`."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = NanoTabICLv2(**ckpt["model_config"])
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt


def build_model(init: str, size: str = "small", cache_dir: Optional[str] = None) -> tuple:
    """(model, model_config) for `--init`:

    - "official": the official TabICLv2 regression checkpoint (downloaded if needed);
    - "scratch": random init at `size` ("small" or "base" = the official dimensions);
    - a file path: an official checkpoint (has "config") or one of ours (has "model_config").
    """
    if init == "official":
        init = download_official(cache_dir=cache_dir)
    if init == "scratch":
        config = dict(SIZES[size])
        return NanoTabICLv2(**config), config
    ckpt = torch.load(init, map_location="cpu", weights_only=False)
    if "model_config" in ckpt:
        model = NanoTabICLv2(**ckpt["model_config"])
        model.load_state_dict(ckpt["state_dict"])
        return model, ckpt["model_config"]
    return load_official(init), nano_config_from_official(ckpt["config"])


SIZES = {
    # nanotabicl README's small regression model
    "small": dict(max_classes=0, out_dim=999, embed_dim=96, col_num_blocks=2, row_num_blocks=2, icl_num_blocks=4,
                  col_nhead=4, row_nhead=4, icl_nhead=4),
    # official TabICLv2 dimensions
    "base": dict(max_classes=0, out_dim=999),
}


if __name__ == "__main__":
    # Download (if needed) + convert the official regression checkpoint and run one forward pass:
    #   python -m nanotabicl.checkpoint [path/to/official.ckpt]
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else download_official()
    model = load_official(path).eval()
    x, y = torch.randn(1, 60, 5), torch.randn(1, 50)
    with torch.no_grad():
        out = model(x, y)
    print(f"{path}: converted OK, {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters, "
          f"output {tuple(out.shape)}, finite={bool(torch.isfinite(out).all())}")
