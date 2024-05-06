import re
from typing import List
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.distributions.normal import Normal


class PlantTraitDataset(Dataset):
    def __init__(
        self, csv, data_dir, sampled_label: bool = False, label_list: List = None
    ) -> None:
        super().__init__()
        self.df = pd.read_csv(csv)

        self.data_dir = Path(data_dir)
        self.sampled_label = sampled_label

        nonid_columns = self.df.columns[1:]
        if label_list is not None:
            self.label_mean_columns = sorted([f"{l}_mean" for l in label_list])
            self.label_sd_columns = sorted([f"{l}_sd" for l in label_list])
        else:
            self.label_mean_columns = sorted(
                [c for c in nonid_columns if re.match(r"^X\d+_mean$", c) is not None]
            )
            self.label_sd_columns = sorted(
                [c for c in nonid_columns if re.match(r"^X\d+_sd$", c) is not None]
            )
        self.ancillary_columns = sorted(
            [c for c in nonid_columns if not c.startswith("X")]
        )

    def cal_ancillary_stats(self):
        values = self.df.loc[:, self.ancillary_columns].values
        return values.mean(axis=0), values.std(axis=0)

    def cal_label_stats(self):
        values = self.df.loc[:, self.label_mean_columns].values
        return values.mean(axis=0), values.std(axis=0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        idx = self.df["id"].iloc[index]
        img_path = self.data_dir / f"{idx}.jpeg"

        ancillaries = [self.df[c].iloc[index] for c in self.ancillary_columns]
        label_means = [self.df[c].iloc[index] for c in self.label_mean_columns]
        label_sds = [self.df[c].iloc[index] for c in self.label_sd_columns]

        if self.sampled_label:
            labels = []
            for mean, sd in zip(label_means, label_sds):
                if not np.isnan(sd):
                    normal = Normal(mean, sd)
                    sampled = normal.sample()
                    labels.append(sampled)
                else:
                    labels.append(mean)
        else:
            labels = label_means

        return {
            "id": int(idx),
            "img_path": str(img_path),
            "ancillaries": np.array(ancillaries, dtype=np.float32),
            "labels": np.array(labels, dtype=np.float32),
            "label_names": self.label_mean_columns,
        }
