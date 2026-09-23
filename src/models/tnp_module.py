from typing import Any, Dict

import torch
from lightning import LightningModule
from torchmetrics import MeanMetric, MinMetric

from src.data.components.crop_vocab import MODALITIES
from src.models.components.tnpd import Batch


class TNPLitModule(LightningModule):
    """`LightningModule` for the Transformer Neural Process (TNP-D) crop model.

    Each batch is one `Batch` episode (context/target split of weather+soil
    tokens); the net predicts a Gaussian per target token and is scored with
    Gaussian negative log-likelihood -- see `src.models.components.tnpd.TNPD`.
    Yield and phenology targets share the same net; they are only
    distinguished by modality id, never by a separate head.

    Besides the loss, every step logs the NLL per target modality (e.g.
    `train/nll_yield`) and, for padded crop-episode batches, the token
    counts (`tokens/context_max`, `tokens/target_max`, `tokens/total_max`),
    so a growing episode size shows up before it becomes an OOM.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.net = net

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # negative log-likelihood is minimized, so track the minimum seen so far
        self.val_loss_best = MinMetric()

    def forward(self, batch: Batch) -> Dict[str, torch.Tensor]:
        return self.net(batch)

    @staticmethod
    def _to_batch(batch: Dict[str, torch.Tensor]) -> Batch:
        # DataLoaders hand us plain tensor dicts (device transfer, collation and
        # pin_memory all have first-class support for those); build the typed
        # `Batch` the net expects right before the forward pass. Extra keys
        # (token counts, episode info) are ignored.
        return Batch.from_dict(batch)

    def on_train_start(self) -> None:
        self.val_loss.reset()
        self.val_loss_best.reset()

    def model_step(self, batch: Dict[str, torch.Tensor], stage: str = "train") -> torch.Tensor:
        typed = self._to_batch(batch)
        outs = self.forward(typed)
        self._log_details(batch, typed, outs, stage)
        return outs["loss"]

    def _log_details(self, batch: Dict[str, Any], typed: Batch, outs: Dict[str, Any], stage: str) -> None:
        bs = typed.xt.shape[0]
        if "num_ctx" in batch:
            num_ctx, num_tar = batch["num_ctx"].float(), batch["num_tar"].float()
            self.log(f"tokens/{stage}_context_max", num_ctx.max(), batch_size=bs)
            self.log(f"tokens/{stage}_target_max", num_tar.max(), batch_size=bs)
            self.log(f"tokens/{stage}_total_max", (num_ctx + num_tar).max(), batch_size=bs, prog_bar=stage == "train")
        ll = outs.get("tar_ll_tokens")
        if ll is None:
            return
        valid = torch.ones_like(ll, dtype=torch.bool) if typed.mask_t is None else typed.mask_t
        for modality_id in torch.unique(typed.mt[valid]).tolist():
            sel = valid & (typed.mt == modality_id)
            name = MODALITIES[modality_id] if modality_id < len(MODALITIES) else str(modality_id)
            self.log(f"{stage}/nll_{name}", -ll[sel].mean(), on_step=False, on_epoch=True, batch_size=bs)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self.model_step(batch, "train")
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch["xt"].shape[0])
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        loss = self.model_step(batch, "val")
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch["xt"].shape[0])

    def on_validation_epoch_end(self) -> None:
        loss = self.val_loss.compute()
        self.val_loss_best(loss)
        self.log("val/loss_best", self.val_loss_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        loss = self.model_step(batch, "test")
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch["xt"].shape[0])

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
