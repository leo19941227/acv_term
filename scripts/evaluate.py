import logging
from typing import List, Any
from collections import defaultdict

import torch
import torchvision
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor

from acv_term.datasets import PlantTraitDataset
from acv_term.models import PlantTraitByUpstream
from acv_term.utils import initialize, get_trainer, get_tester, find_last_ckpt


class Collater:
    def __init__(self, upstream_name: str) -> None:
        self.processor = AutoImageProcessor.from_pretrained(upstream_name)

    def __call__(self, samples: List[Any]):
        collated = defaultdict(list)
        for sample in samples:
            collated["id"].append(sample["id"])

            image = torchvision.io.read_image(sample["img_path"])
            image = self.processor(image, return_tensors="pt")["pixel_values"][0]
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
            "label_names": samples[0]["label_names"],
        }


def make_valid_dataloader(conf):
    valid_dataset = PlantTraitDataset(**conf["data"]["valid"])

    def make_tensor(arrays):
        for a in arrays:
            yield torch.FloatTensor(a)

    # calculate data statistics
    ancillaries_mean, ancillaries_std = make_tensor(valid_dataset.cal_ancillary_stats())
    labels_mean, labels_std = make_tensor(valid_dataset.cal_label_stats())

    valid_dataloader = DataLoader(
        valid_dataset,
        **conf["dataloader"]["valid"],
        collate_fn=Collater(**conf["collater"]),
    )
    return valid_dataloader, ancillaries_mean, ancillaries_std, labels_mean, labels_std


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    conf = initialize()

    valid_dataloader, ancillaries_mean, ancillaries_std, labels_mean, labels_std = (
        make_valid_dataloader(conf)
    )
    model = PlantTraitByUpstream.load_from_checkpoint(
        conf["ckpt"],
        **conf["model"],
        ancillaries_mean=ancillaries_mean,
        ancillaries_std=ancillaries_std,
        labels_mean=labels_mean,
        labels_std=labels_std,
    )

    tester = get_tester(conf["expdir"], conf["trainer"])
    tester.test(model=model, dataloaders=valid_dataloader)
