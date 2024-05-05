from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.distributions.normal import Normal


class PlantTraitDataset(Dataset):
    def __init__(self, csv, data_dir, sampled_label: bool = False) -> None:
        super().__init__()
        self.df = pd.read_csv(csv)

        self.data_dir = Path(data_dir)
        self.sampled_label = sampled_label

        self.ancillary_columns = self.df.columns[1:-12]
        self.label_mean_columns = self.df.columns[-12:-6]
        self.label_sd_columns = self.df.columns[-6:]
        for column in self.label_mean_columns:
            assert column.startswith("X") and "mean" in column
        for column in self.label_sd_columns:
            assert column.startswith("X") and "sd" in column

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
        }
