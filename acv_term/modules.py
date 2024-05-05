from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PredictionHead(nn.Module):
    def __init__(
        self,
        num_input_layers: int,
        input_size: int,
        hidden_size: int,
        num_layer: int = 2,
    ) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(num_input_layers))
        module_list = [nn.Linear(2 * input_size, hidden_size), nn.GELU()]
        for _ in range(num_layer - 1):
            module_list.extend(
                [
                    nn.Linear(hidden_size, hidden_size),
                    nn.GELU(),
                ]
            )
        module_list.append(nn.Linear(hidden_size, 1))
        self.proj = nn.Sequential(*module_list)

    def forward(self, input_hidden_states: Tuple[torch.FloatTensor]):
        weights = F.softmax(self.weights, dim=0)
        input_hidden_states = torch.stack(input_hidden_states, dim=0)
        hs = (weights.view(-1, 1, 1, 1) * input_hidden_states).sum(dim=0)
        # (batch_size, num_patch, input_size)

        # statistic pooling
        mean = hs.mean(dim=1)
        std = hs.std(dim=1)
        h = torch.cat([mean, std], dim=1)
        pred = self.proj(h)
        return pred.reshape(len(hs))  # (batch_size, )


class DownstreamModel(nn.Module):
    def __init__(
        self,
        num_ancillary: int,
        upstream_size: int,
        upstream_num_layer: int,
        hidden_size: int,
        num_head: int,
        num_layer: int,
        num_predict_head: int,
        ancillaries_mean: torch.FloatTensor,
        ancillaries_std: torch.FloatTensor,
    ) -> None:
        super().__init__()
        self.register_buffer("ancillaries_mean", ancillaries_mean)
        self.register_buffer("ancillaries_std", ancillaries_std)

        self.layer_projs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(upstream_size, hidden_size),
                    nn.GELU(),
                    nn.Linear(hidden_size, hidden_size),
                )
                for _ in range(upstream_num_layer)
            ]
        )

        self.ancillaries_proj = nn.Sequential(
            nn.Linear(num_ancillary, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.pooler_proj = nn.Sequential(
            nn.Linear(upstream_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.encoder_blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    hidden_size,
                    num_head,
                    hidden_size * 3,
                    batch_first=True,
                )
                for _ in range(num_layer)
            ]
        )
        self.predict_heads = nn.ModuleList(
            [
                PredictionHead(
                    num_layer + 1,
                    hidden_size,
                    hidden_size,
                )
                for _ in range(num_predict_head)
            ]
        )

    def forward(
        self,
        hidden_states: Tuple[torch.FloatTensor],
        ancillaries: torch.FloatTensor,
        pooler_output: torch.FloatTensor = None,
    ):
        projs = []
        for hs, proj in zip(hidden_states, self.layer_projs):
            hs = F.layer_norm(hs, hs.shape[-1:])
            hs = proj(hs)
            projs.append(hs)
        reduced_hs = torch.stack(projs, dim=0).mean(
            dim=0
        )  # (batch_size, num_patch, hidden_size)

        # incoporating global information
        ancillaries = (ancillaries - self.ancillaries_mean) / (
            self.ancillaries_std + 1.0e-8
        )
        ancillaries = self.ancillaries_proj(ancillaries)
        hs = reduced_hs + ancillaries.unsqueeze(1)

        if pooler_output is not None:
            pooler_output = F.layer_norm(pooler_output, pooler_output.shape[-1:])
            pooler_output = self.pooler_proj(pooler_output)
            hs = hs + pooler_output.unsqueeze(1)

        # add positional embedding

        # encode
        all_hs = [hs]
        for block in self.encoder_blocks:
            hs = block(hs)
            all_hs.append(hs)

        all_preds = []
        for pred in self.predict_heads:
            pred = pred(all_hs)  # (batch_size, )
            all_preds.append(pred)
        all_preds = torch.stack(all_preds, 1)  # (batch_size, num_label)

        return all_preds
