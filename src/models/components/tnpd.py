from dataclasses import dataclass, fields
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions.normal import Normal

from src.models.components.tnp_modules import CoordinateEncoder, ModalityEmbedding, TNPEncoderLayer, build_mlp


@dataclass
class Batch:
    """One batch of neural-process episodes, split into context and target sets.

    Coordinates carry `(lat, lon, t, depth)` per token; modality ids index
    into the shared modality-identity lookup table (e.g. one id for "tmax",
    another for "clay"); values are scalars (z-scored). Cultivar/sowing-date
    are intentionally never tokens here -- they only vary how an episode was
    simulated, per the Phase 1 calibration design.

    xc: [B, Nc, 4]   yc: [B, Nc, 1]   mc: [B, Nc] (long)   mask_c: [B, Nc] (bool)
    xt: [B, Nt, 4]   yt: [B, Nt, 1]   mt: [B, Nt] (long)   mask_t: [B, Nt] (bool)

    `mask_c`/`mask_t` are True for real tokens, False for padding (see
    `collate_episodes`); None means no padding.
    """

    xc: torch.Tensor
    yc: torch.Tensor
    mc: torch.Tensor
    xt: torch.Tensor
    yt: torch.Tensor
    mt: torch.Tensor
    mask_c: Optional[torch.Tensor] = None
    mask_t: Optional[torch.Tensor] = None

    @classmethod
    def from_dict(cls, batch: dict) -> "Batch":
        """Build from a collated dict, ignoring its extra keys (token counts, info)."""
        return cls(**{f.name: batch[f.name] for f in fields(cls) if f.name in batch})


class TNP(nn.Module):
    """Transformer Neural Process encoder.

    Adapted from the official TNP-pytorch implementation
    (tung-nd/TNP-pytorch, `regression/models/tnp.py`):

    - tokens are `(coordinate, modality_id, value)` instead of `(x, y)`: the
      embedder input is `concat(fourier(coordinate), modality_embedding,
      value)`;
    - a target token's value is replaced by a learnable mask value (the
      original feeds zeros; initialised at zero, so it starts identical);
    - the attention pattern is the original's (context tokens attend to all
      context tokens, target tokens attend to context tokens only), run as a
      context stream plus a target stream sharing each layer's weights
      (`TNPEncoderLayer`), which is mathematically the same as the masked
      joint pass but only needs a padding mask over context keys;
    - padding: padded context positions are never attended to.
    """

    def __init__(
        self,
        dim_y: int,
        d_model: int,
        emb_depth: int,
        dim_feedforward: int,
        nhead: int,
        dropout: float,
        num_layers: int,
        num_modalities: int,
        dim_modality: int = 16,
        num_time_freqs: int = 12,
        num_space_freqs: int = 8,
        num_depth_freqs: int = 4,
    ):
        super().__init__()

        self.coord_encoder = CoordinateEncoder(
            num_time_freqs=num_time_freqs, num_space_freqs=num_space_freqs, num_depth_freqs=num_depth_freqs
        )
        self.modality_embedding = ModalityEmbedding(num_modalities, dim_modality)
        self.mask_value = nn.Parameter(torch.zeros(dim_y))

        dim_token_in = self.coord_encoder.out_dim + self.modality_embedding.out_dim + dim_y
        self.embedder = build_mlp(dim_token_in, d_model, d_model, emb_depth)

        self.layers = nn.ModuleList(
            [TNPEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True) for _ in range(num_layers)]
        )

    def tokenize(self, x: torch.Tensor, y: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        coord_enc = self.coord_encoder(x)
        mod_emb = self.modality_embedding(m)
        return torch.cat([coord_enc, mod_emb, y], dim=-1)

    def construct_input(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        tok_ctx = self.tokenize(batch.xc, batch.yc, batch.mc)
        tok_tar = self.tokenize(batch.xt, self.mask_value.expand_as(batch.yt), batch.mt)
        return tok_ctx, tok_tar

    def create_mask(self, batch: Batch) -> Optional[torch.Tensor]:
        """Key padding mask over context tokens: True = padding, never attended to.
        The context/target structure itself is enforced by the two streams."""
        return None if batch.mask_c is None else ~batch.mask_c

    def encode(self, batch: Batch) -> torch.Tensor:
        tok_ctx, tok_tar = self.construct_input(batch)
        padding = self.create_mask(batch)
        ctx, tar = self.embedder(tok_ctx), self.embedder(tok_tar)
        for layer in self.layers:
            ctx, tar = layer(ctx, ctx, padding), layer(tar, ctx, padding)
        return tar


class TNPD(TNP):
    """TNP-D: deterministic decoder head on top of the TNP encoder.

    Predicts a diagonal Gaussian per target token; scored with Gaussian
    log-likelihood. No task-specific heads -- yield and phenology are just
    different target tokens (different modality ids) queried from the same
    decoder, per the Phase 1 training objective. The loss is the mean
    negative log-likelihood over all real (non-padded) target tokens of the
    batch, all target types jointly.
    """

    def __init__(self, *args, bound_std: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        d_model = self.embedder[-1].out_features

        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2),
        )
        self.bound_std = bound_std

    def _predict_normal(self, z_target: torch.Tensor) -> Normal:
        out = self.predictor(z_target)
        mean, std = torch.chunk(out, 2, dim=-1)
        if self.bound_std:
            std = 0.05 + 0.95 * F.softplus(std)
        else:
            std = torch.exp(std)
        return Normal(mean, std)

    def forward(self, batch: Batch, reduce_ll: bool = True) -> dict:
        z_target = self.encode(batch)
        pred_tar = self._predict_normal(z_target)

        ll = pred_tar.log_prob(batch.yt).sum(-1)  # [B, Nt]
        valid = torch.ones_like(ll, dtype=torch.bool) if batch.mask_t is None else batch.mask_t
        # zero padded targets before reducing (not after), so they contribute exactly nothing
        ll = torch.where(valid, ll, torch.zeros_like(ll))

        if reduce_ll:
            tar_ll = ll.sum() / valid.sum().clamp(min=1)
        else:
            tar_ll = ll

        return {"loss": -tar_ll if reduce_ll else -ll, "tar_ll": tar_ll, "tar_ll_tokens": ll, "pred_tar": pred_tar}

    @torch.no_grad()
    def predict(
        self,
        xc: torch.Tensor,
        yc: torch.Tensor,
        mc: torch.Tensor,
        xt: torch.Tensor,
        mt: torch.Tensor,
        mask_c: Optional[torch.Tensor] = None,
    ) -> Normal:
        yt_dummy = torch.zeros((xt.shape[0], xt.shape[1], yc.shape[2]), device=xt.device)
        batch = Batch(xc=xc, yc=yc, mc=mc, xt=xt, yt=yt_dummy, mt=mt, mask_c=mask_c)
        z_target = self.encode(batch)
        return self._predict_normal(z_target)
