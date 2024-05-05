from pathlib import Path

import numpy as np

from acv_term.datasets import PlantTraitDataset


def test_PlantTraitDataset():
    csv = "/home/leo1994122701/acv/acv_term/data/split_9_1_seed0/train.csv"
    data_dir = "/home/leo1994122701/acv/data/train_images"

    dataset = PlantTraitDataset(csv, data_dir)
    item = dataset[0]

    assert isinstance(item["id"], int)
    assert Path(item["img_path"]).is_file()
    assert isinstance(item["ancillaries"][0], np.float32)
    assert isinstance(item["labels"][0], np.float32)
