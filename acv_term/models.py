from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl
import torch.nn.functional as F
from transformers import AutoModel
from sklearn.metrics import r2_score

from acv_term.modules import DownstreamModel


class PlantTraitByUpstream(pl.LightningModule):
    def __init__(
        self,
        upstream_name: str,
        num_ancillary: int,
        hidden_size: int,
        num_head: int,
        num_layer: int,
        num_predict_head: int,
        upstream_trainable: bool,
        ancillaries_mean: torch.FloatTensor,
        ancillaries_std: torch.FloatTensor,
        labels_mean: torch.FloatTensor,
        labels_std: torch.FloatTensor,
        lr: float = 1.0e-3,
    ) -> None:
        super().__init__()
        self.upstream: nn.Module = AutoModel.from_pretrained(upstream_name)
        if not upstream_trainable:
            self.upstream.requires_grad_(False)

        self.upstream_trainable = upstream_trainable
        self.lr = lr

        self.downstream = DownstreamModel(
            num_ancillary,
            self.upstream.config.hidden_size,
            self.upstream.config.num_hidden_layers,
            hidden_size,
            num_head,
            num_layer,
            num_predict_head,
            ancillaries_mean,
            ancillaries_std,
        )

        self.register_buffer("labels_mean", labels_mean)
        self.register_buffer("labels_std", labels_std)
        self.valid_records = defaultdict(list)

    def forward(self, images: torch.FloatTensor, ancillaries: torch.FloatTensor):
        with torch.set_grad_enabled(self.upstream_trainable):
            if self.upstream_trainable:
                self.upstream.train()
            else:
                self.upstream.eval()

            results = self.upstream(
                pixel_values=images, output_hidden_states=True, return_dict=True
            )  # (batch_size, num_patch, hidden_size)

        hidden_states = results[
            "hidden_states"
        ]  # tuple of (batch_size, num_patch, hidden_size)
        pooler_output = results.get("pooler_output", None)  # (batch_size, hidden_size)

        preds = self.downstream(
            hidden_states,
            ancillaries,
            pooler_output,
        )  # (batch_size, num_head)
        return preds

    @staticmethod
    def compute_r2(preds: torch.FloatTensor, labels: torch.FloatTensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        r2 = r2_score(labels, preds, multioutput="uniform_average")
        return r2

    def training_step(self, batch, batch_idx):
        self.log("global_step", self.global_step, on_step=True)

        images = batch["images"]  # (batch_size, 3, 224, 224)
        ancillaries = batch["ancillaries"]  # (batch_size, num_anc)
        labels = batch["labels"]  # (batch_size, num_label)
        bs = len(images)

        preds = self(images, ancillaries)
        norm_labels = (labels - self.labels_mean) / (self.labels_std + 1.0e-8)

        mse = F.mse_loss(preds, norm_labels, reduction="none")
        loss = mse.mean(dim=1).mean(dim=0)

        self.log(f"train/loss", loss, on_step=True, prog_bar=True, batch_size=bs)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch["images"]  # (batch_size, 3, 224, 224)
        ancillaries = batch["ancillaries"]  # (batch_size, num_anc)
        labels = batch["labels"]  # (batch_size, num_label)

        preds = self(images, ancillaries)
        scale_preds = (preds * self.labels_std) + self.labels_mean

        self.valid_records["labels"].extend(labels.detach().cpu().unbind(dim=0))
        self.valid_records["predicts"].extend(scale_preds.detach().cpu().unbind(dim=0))

    def on_validation_epoch_end(self) -> None:
        labels = torch.stack(self.valid_records["labels"], dim=0)
        predicts = torch.stack(self.valid_records["predicts"], dim=0)
        self.valid_records = defaultdict(list)

        r2 = r2_score(labels, predicts)

        self.log(f"valid/r2", r2, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
