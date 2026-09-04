from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions.normal import Normal

from src.models.components.tnp_modules import build_mlp, CoordinateEncoder, ModalityEmbedding


@dataclass
class Batch:
    """One neural-process episode, split into context and target sets.

    Coordinates carry `(lat, lon, time, depth)` per token; modality ids index
    into the shared modality-identity lookup table (e.g. one id for "TMAX",
    another for "clay %"); values are scalars. Cultivar/sowing-date are
    intentionally never tokens here -- they only vary how an episode was
    simulated, per the Phase 1 calibration design.

    xc: [B, Nc, 4]   yc: [B, Nc, 1]   mc: [B, Nc] (long)
    xt: [B, Nt, 4]   yt: [B, Nt, 1]   mt: [B, Nt] (long)
    """

    xc: torch.Tensor
    yc: torch.Tensor
    mc: torch.Tensor
    xt: torch.Tensor
    yt: torch.Tensor
    mt: torch.Tensor


class TNP(nn.Module):
    """Transformer Neural Process encoder.

    Adapted from the official TNP-pytorch implementation
    (tung-nd/TNP-pytorch, `regression/models/tnp.py`). The only structural
    change from the original is widening `construct_input` so each token
    carries modality-identity alongside its coordinate and value, per the
    Phase 1 architecture spec -- everything else (masked self-attention over
    context+target in one pass, context tokens fully visible, target tokens
    only attending to context) is unchanged. The original's hardcoded
    `device='cuda'` in `create_mask` is also fixed to follow the input tensors,
    so this runs on CPU/MPS/GPU alike.
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
        num_time_freqs: int = 8,
    ):
        super().__init__()

        self.coord_encoder = CoordinateEncoder(num_time_freqs=num_time_freqs)
        self.modality_embedding = ModalityEmbedding(num_modalities, dim_modality)

        dim_token_in = self.coord_encoder.out_dim + self.modality_embedding.out_dim + dim_y
        self.embedder = build_mlp(dim_token_in, d_model, d_model, emb_depth)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def tokenize(self, x: torch.Tensor, y: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        coord_enc = self.coord_encoder(x)
        mod_emb = self.modality_embedding(m)
        return torch.cat([coord_enc, mod_emb, y], dim=-1)

    def construct_input(self, batch: Batch) -> torch.Tensor:
        tok_ctx = self.tokenize(batch.xc, batch.yc, batch.mc)
        tok_tar = self.tokenize(batch.xt, torch.zeros_like(batch.yt), batch.mt)
        return torch.cat([tok_ctx, tok_tar], dim=1)

    def create_mask(self, batch: Batch) -> tuple[torch.Tensor, int]:
        num_ctx = batch.xc.shape[1]
        num_tar = batch.xt.shape[1]
        num_all = num_ctx + num_tar
        mask = torch.zeros(num_all, num_all, device=batch.xc.device).fill_(float("-inf"))
        mask[:, :num_ctx] = 0.0
        return mask, num_tar

    def encode(self, batch: Batch) -> torch.Tensor:
        inp = self.construct_input(batch)
        mask, num_tar = self.create_mask(batch)
        embeddings = self.embedder(inp)
        out = self.encoder(embeddings, mask=mask)
        return out[:, -num_tar:]


class TNPD(TNP):
    """TNP-D: deterministic decoder head on top of the TNP encoder.

    Predicts a diagonal Gaussian per target token; scored with Gaussian
    log-likelihood. No task-specific heads -- yield and phenology are just
    different target tokens (different modality ids) queried from the same
    decoder, per the Phase 1 training objective.
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

        if reduce_ll:
            tar_ll = pred_tar.log_prob(batch.yt).sum(-1).mean()
        else:
            tar_ll = pred_tar.log_prob(batch.yt).sum(-1)

        return {"loss": -tar_ll, "tar_ll": tar_ll, "pred_tar": pred_tar}

    @torch.no_grad()
    def predict(
        self,
        xc: torch.Tensor,
        yc: torch.Tensor,
        mc: torch.Tensor,
        xt: torch.Tensor,
        mt: torch.Tensor,
    ) -> Normal:
        yt_dummy = torch.zeros((xt.shape[0], xt.shape[1], yc.shape[2]), device=xt.device)
        batch = Batch(xc=xc, yc=yc, mc=mc, xt=xt, yt=yt_dummy, mt=mt)
        z_target = self.encode(batch)
        return self._predict_normal(z_target)
