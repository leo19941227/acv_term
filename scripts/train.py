import logging
from typing import List, Any
from collections import defaultdict

import torch
import torchvision
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModel

from acv_term.datasets import PlantTraitDataset
from acv_term.utils import initialize, get_trainer, find_last_ckpt, merge_list_of_dict


class Collater:
    def __init__(self, upstream_name: str) -> None:
        self.processor = AutoImageProcessor.from_pretrained(upstream_name)

    def __call__(self, samples: List[Any]):
        collated = defaultdict(list)
        for sample in samples:
            collated["id"].append(sample["id"])

            image = torchvision.io.read_image(sample["img_path"])
            image = self.processor(image)["pixel_values"][0]
            collated["images"].append(torch.FloatTensor(image))

            collated["ancillaries"].append(torch.FloatTensor(sample["ancillaries"]))
            collated["labels"].append(torch.FloatTensor(sample["labels"]))

        return {
            "id": torch.LongTensor(collated["id"]),  # (batch_size, )
            "images": torch.stack(
                collated["images"], dim=0
            ),  # (batch_size, 3, 224, 224)
            "ancillaries": torch.stack(
                collated["ancillaries"], dim=0
            ),  # (batch_size, num_ancillaries)
            "labels": torch.stack(
                collated["labels"], dim=0
            ),  # (batch_size, num_labels)
        }


def make_dataloaders(conf):
    train_dataset = PlantTraitDataset(**conf["data"]["train"])
    valid_dataset = PlantTraitDataset(**conf["data"]["valid"])

    train_dataloader = DataLoader(
        train_dataset,
        **conf["dataloader"]["train"],
        collate_fn=Collater(**conf["collater"]),
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        **conf["dataloader"]["valid"],
        collate_fn=Collater(**conf["collater"]),
    )
    return train_dataloader, valid_dataloader


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    conf = initialize()

    train_dataloader, valid_dataloader = make_dataloaders(conf)
    model = UnconditionalBarkSemantic(**conf["model"])

    trainer = get_trainer(conf["expdir"], conf["trainer"], **conf["checkpoint"])
    last_ckpt = find_last_ckpt(conf["expdir"])
    trainer.fit(
        model=model,
        ckpt_path=last_ckpt,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )
