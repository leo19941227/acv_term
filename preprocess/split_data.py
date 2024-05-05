import random
import logging
import argparse
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("output_dir")
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)

    df = pd.read_csv(args.csv)
    indices = list(range(len(df)))
    random.shuffle(indices)
    valid_num = round(len(indices) * args.valid_ratio)

    train_indices = indices[:-valid_num]
    valid_indices = indices[-valid_num:]

    train_df = df.iloc[train_indices]
    valid_df = df.iloc[valid_indices]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    train_df.to_csv(str(output_dir / "train.csv"), index=None)
    valid_df.to_csv(str(output_dir / "valid.csv"), index=None)
