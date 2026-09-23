import math
from typing import Optional

import torch
from torch import nn


def build_mlp(dim_in: int, dim_hid: int, dim_out: int, depth: int) -> nn.Sequential:
    """MLP builder, ported as-is from the official TNP-pytorch `models/modules.py`."""
    modules = [nn.Linear(dim_in, dim_hid), nn.ReLU(True)]
    for _ in range(depth - 2):
        modules.append(nn.Linear(dim_hid, dim_hid))
        modules.append(nn.ReLU(True))
    modules.append(nn.Linear(dim_hid, dim_out))
    return nn.Sequential(*modules)


def _log_spaced_periods(min_period: float, max_period: float, num: int) -> torch.Tensor:
    if num == 1:
        return torch.tensor([max_period])
    return torch.logspace(math.log10(min_period), math.log10(max_period), num)


class CoordinateEncoder(nn.Module):
    """Sinusoidal (Fourier) encoding of a `(lat, lon, t, depth)` coordinate.

    Each dimension gets `sin`/`cos` features at log-spaced periods covering
    its natural scales, plus the raw (scaled) value:

    - `t`: a continuous timestamp (days since a fixed reference date, not a
      sequence index or day-of-year rank), periods from `min_time_period`
      (sub-weekly) to `max_time_period` (multi-year) -- so weekly, dekadal
      and irregular timesteps are all handled by the same encoding;
    - `lat`/`lon` (degrees): periods from `min_space_period` to 360;
    - `depth` (cm, 0 for non-soil tokens): periods from `min_depth_period`
      to `max_depth_period`.
    """

    def __init__(
        self,
        num_time_freqs: int = 12,
        min_time_period: float = 2.0,
        max_time_period: float = 365.25 * 32,
        num_space_freqs: int = 8,
        min_space_period: float = 0.5,
        max_space_period: float = 360.0,
        num_depth_freqs: int = 4,
        min_depth_period: float = 5.0,
        max_depth_period: float = 400.0,
    ):
        super().__init__()
        self.register_buffer("time_freqs", 2 * math.pi / _log_spaced_periods(min_time_period, max_time_period, num_time_freqs))
        self.register_buffer("space_freqs", 2 * math.pi / _log_spaced_periods(min_space_period, max_space_period, num_space_freqs))
        self.register_buffer("depth_freqs", 2 * math.pi / _log_spaced_periods(min_depth_period, max_depth_period, num_depth_freqs))
        # raw coordinates, scaled to roughly [-1, 1]
        self.register_buffer("raw_scale", torch.tensor([1 / 90.0, 1 / 180.0, 1 / max_time_period, 1 / max_depth_period]))

    @property
    def out_dim(self) -> int:
        return 4 + 2 * (2 * len(self.space_freqs) + len(self.time_freqs) + len(self.depth_freqs))

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """:param coords: [..., 4] tensor of (lat, lon, t, depth)."""
        lat, lon, t, depth = coords.unbind(dim=-1)
        angles = torch.cat(
            [
                lat.unsqueeze(-1) * self.space_freqs,
                lon.unsqueeze(-1) * self.space_freqs,
                t.unsqueeze(-1) * self.time_freqs,
                depth.unsqueeze(-1) * self.depth_freqs,
            ],
            dim=-1,
        )
        return torch.cat([coords * self.raw_scale, torch.sin(angles), torch.cos(angles)], dim=-1)


class ModalityEmbedding(nn.Module):
    """Learnable lookup table for modality-identity, one row per known variable
    (max temp, precip, clay%, ...). Open-vocabulary/metadata-derived modality
    embeddings are a later upgrade, not needed for Phase 1.
    """

    def __init__(self, num_modalities: int, dim_modality: int):
        super().__init__()
        self.embedding = nn.Embedding(num_modalities, dim_modality)

    @property
    def out_dim(self) -> int:
        return self.embedding.embedding_dim

    def forward(self, modality_ids: torch.Tensor) -> torch.Tensor:
        """:param modality_ids: [...] long tensor of modality indices."""
        return self.embedding(modality_ids)


class TNPEncoderLayer(nn.TransformerEncoderLayer):
    """`nn.TransformerEncoderLayer` whose queries attend to a separate
    key/value set.

    The TNP attention mask lets context tokens attend to context only and
    target tokens attend to context only. So context never depends on the
    targets, and one masked pass over `[context; targets]` equals, layer by
    layer:

        context <- layer(context, attending to context)
        targets <- layer(targets, attending to context)

    with the same weights. Running it this way needs only a per-key padding
    mask ([B, N_context]) instead of a [N, N] attention mask that PyTorch
    would expand per batch element and head, which matters at thousands of
    tokens. Post-norm, ReLU, dropout: identical to the parent's defaults, and
    the parameters are the parent's, so a stock layer computes the same
    function (see tests/test_tnp_crop.py).
    """

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """:param context_padding_mask: [B, N_context] bool, True = padding (ignored)."""
        context = x if context is None else context
        attn = self.self_attn(x, context, context, key_padding_mask=context_padding_mask, need_weights=False)[0]
        if self.norm_first:
            raise NotImplementedError("TNPEncoderLayer mirrors the post-norm TNP setup only.")
        x = self.norm1(x + self.dropout1(attn))
        return self.norm2(x + self._ff_block(x))
