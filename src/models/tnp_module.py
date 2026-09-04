from typing import Any, Dict

import torch
from lightning import LightningModule
from torchmetrics import MeanMetric, MinMetric

from src.models.components.tnpd import Batch


class TNPLitModule(LightningModule):
    """`LightningModule` for the Transformer Neural Process (TNP-D) crop model.

    Each batch is one `Batch` episode (context/target split of weather+soil
    tokens); the net predicts a Gaussian per target token and is scored with
    Gaussian negative log-likelihood -- see `src.models.components.tnpd.TNPD`.
    Yield and phenology targets share the same net; they are only
    distinguished by modality id, never by a separate head.
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
        # `Batch` the net expects right before the forward pass.
        return Batch(**batch)

    def on_train_start(self) -> None:
        self.val_loss.reset()
        self.val_loss_best.reset()

    def model_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        outs = self.forward(self._to_batch(batch))
        return outs["loss"]

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self.model_step(batch)
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        loss = self.model_step(batch)
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        loss = self.val_loss.compute()
        self.val_loss_best(loss)
        self.log("val/loss_best", self.val_loss_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        loss = self.model_step(batch)
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

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
