import math

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


class CoordinateEncoder(nn.Module):
    """Encodes a `(lat, lon, time, depth)` coordinate into a fixed-size vector.

    Per the Phase 1 architecture spec: time uses a timestamp-based sinusoidal
    (Fourier) encoding -- the actual date fed into the formula, not a
    sequence-rank/lookup encoding -- so mixed timestep densities (daily,
    weekly, dekadal, irregular) are handled by the same model. `time` is
    expected as a plain float (e.g. days since some fixed reference epoch);
    lat/lon/depth pass through unchanged.
    """

    def __init__(self, num_time_freqs: int = 8, max_period: float = 365.25):
        super().__init__()
        self.num_time_freqs = num_time_freqs
        freqs = torch.tensor(
            [2 * math.pi / max_period * (2**i) for i in range(num_time_freqs)],
            dtype=torch.float32,
        )
        self.register_buffer("freqs", freqs)

    @property
    def out_dim(self) -> int:
        # lat, lon, depth + sin/cos pairs per time frequency
        return 3 + 2 * self.num_time_freqs

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """:param coords: [..., 4] tensor of (lat, lon, time, depth)."""
        lat, lon, time, depth = coords.unbind(dim=-1)
        angles = time.unsqueeze(-1) * self.freqs
        time_enc = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return torch.cat(
            [lat.unsqueeze(-1), lon.unsqueeze(-1), depth.unsqueeze(-1), time_enc], dim=-1
        )


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
