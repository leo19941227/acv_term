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
        use_ancillaries: bool = True,
        norm_label: bool = False,
        lr: float = 1.0e-3,
    ) -> None:
        super().__init__()
        self.upstream: nn.Module = AutoModel.from_pretrained(upstream_name)
        if not upstream_trainable:
            self.upstream.requires_grad_(False)

        self.upstream_trainable = upstream_trainable
        self.norm_label = norm_label
        self.lr = lr

        self.downstream = DownstreamModel(
            num_ancillary,
            self.upstream.config.hidden_size,
            self.upstream.config.num_hidden_layers,
            hidden_size,
            num_head,
            num_layer,
            num_predict_head,
            use_ancillaries,
            ancillaries_mean,
            ancillaries_std,
        )

        self.register_buffer("labels_mean", labels_mean.unsqueeze(0))
        self.register_buffer("labels_std", labels_std.unsqueeze(0))
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
        r2 = r2_score(labels, preds, multioutput="raw_values")
        r2 = torch.FloatTensor(r2)
        return r2

    def supervised_step(self, batch, batch_idx):
        images = batch["images"]  # (batch_size, 3, 224, 224)
        ancillaries = batch["ancillaries"]  # (batch_size, num_anc)
        labels = batch["labels"]  # (batch_size, num_label)

        preds = self(images, ancillaries)

        if self.norm_label:
            target_labels = (labels - self.labels_mean) / (self.labels_std + 1.0e-8)
            post_preds = preds * self.labels_std + self.labels_mean
        else:
            target_labels = labels
            post_preds = preds

        train_loss = (
            F.mse_loss(preds, target_labels, reduction="none").mean(dim=0).mean(dim=0)
        )
        losses = F.mse_loss(post_preds, labels, reduction="none").mean(dim=0)
        r2s = self.compute_r2(post_preds, labels)

        return train_loss, losses, r2s, post_preds, labels

    def training_step(self, batch, batch_idx):
        self.log("global_step", self.global_step, on_step=True)

        train_loss, losses, r2s, preds, labels = self.supervised_step(batch, batch_idx)
        bs = len(preds)

        self.log(
            f"train/train_loss", train_loss, on_step=True, prog_bar=True, batch_size=bs
        )

        for loss, r2, name in zip(losses, r2s, batch["label_names"]):
            self.log(f"train/{name}_loss", loss, on_step=True, batch_size=bs)
            self.log(f"train/{name}_r2", r2, on_step=True, batch_size=bs)

        self.log(
            f"train/avg_r2", r2s.mean(), on_step=True, prog_bar=True, batch_size=bs
        )
        return train_loss

    def validation_step(self, batch, batch_idx):
        _, losses, r2s, preds, labels = self.supervised_step(batch, batch_idx)

        self.valid_records["labels"].extend(labels.detach().cpu().unbind(dim=0))
        self.valid_records["predicts"].extend(preds.detach().cpu().unbind(dim=0))
        self.valid_records["label_names"] = batch["label_names"]

    def on_validation_epoch_end(self) -> None:
        labels = torch.stack(self.valid_records["labels"], dim=0)
        predicts = torch.stack(self.valid_records["predicts"], dim=0)
        label_names = self.valid_records["label_names"]
        self.valid_records = defaultdict(list)

        losses = F.mse_loss(predicts, labels, reduction="none").mean(dim=0)
        r2s = self.compute_r2(predicts, labels)

        for loss, r2, name in zip(losses, r2s, label_names):
            self.log(f"valid/{name}_loss", loss, on_epoch=True)
            self.log(f"valid/{name}_r2", r2, on_epoch=True)

        self.log(f"valid/avg_r2", r2s.mean(), on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
