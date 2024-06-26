import os
import re
import sys
import yaml
import random
import logging
import argparse
from time import time
from pathlib import Path
from datetime import datetime
from typing import Any, List, Tuple
from collections import defaultdict
from contextlib import ContextDecorator
from omegaconf import OmegaConf, open_dict

import torch
import numpy as np
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger

logger = logging.getLogger(__name__)


def get_time_tag():
    return datetime.fromtimestamp(time()).strftime("%Y-%m-%d-%H-%M-%S")


def get_masks_from_lengths(lengths: torch.LongTensor):
    assert isinstance(lengths, torch.Tensor)
    device = lengths.device
    return (
        torch.arange(max(lengths)).view(1, -1).repeat(len(lengths), 1).to(device)
        < lengths.unsqueeze(1)
    ).long()


def merge_list_of_dict(list_of_dict: List[dict]) -> dict:
    keys = list(list_of_dict[0].keys())
    merged = defaultdict(list)
    for item in list_of_dict:
        for key in keys:
            merged[key].append(item[key])
    return dict(merged)


def flatten_list_of_list(list_of_list: List[List[Any]]) -> List:
    flattened = []
    for items in list_of_list:
        flattened.extend(items)
    return flattened


def find_last_ckpt(expdir: str):
    expdir = Path(expdir)
    if not expdir.is_dir():
        return None

    files = sorted(os.listdir(expdir))
    files = [file for file in files if file.startswith("global_step")]

    if len(files) == 0:
        return None

    files.sort(
        key=lambda x: int(re.search("global_step-(\d+).ckpt", x).groups()[0]),
        reverse=True,
    )
    file_paths = [str(expdir / file) for file in files]

    last_ckpt = None
    for file_path in file_paths:
        try:
            # check the integrity of the checkpoint
            torch.load(file_path)
        except:
            logger.warning(f"Fail to load {file_path}. Deleting.")
            Path(file_path).unlink()
            continue
        else:
            logger.info(f"Using last checkpoint: {file_path}")
            last_ckpt = file_path
            break
    return last_ckpt


def parse_overrides(options: list):
    """
    Example usgae:
        [
            "--optimizer.lr",
            "1.0e-3",
            "++optimizer.name",
            "AdamW",
            "--runner.eval_dataloaders",
            "['dev', 'test']",
        ]

    Convert to:
        {
            "optimizer": {"lr": 1.0e-3},
            "runner": {"eval_dataloaders": ["dev", "test"]}
        },
        {
            "optimizer": {"name": "AdamW"}
        }
    """
    revise = []
    add = []
    assert len(options) % 2 == 0, "An option must be followed by a value."
    for position in range(0, len(options), 2):
        key = options[position].strip()
        value_str = options[position + 1].strip()
        if key.startswith("--"):
            key = key.strip("--")
            revise.append((key, value_str))
        elif key.startswith("++"):
            key = key.strip("++")
            add.append((key, value_str))
        else:
            raise ValueError("command line argument must start with -- or ++")

    def parse_single_key_value(conf: dict, key: str, value_str: str):
        remaining = key.split(".")
        try:
            value = eval(value_str)
        except:
            value = value_str

        target_conf = conf
        for i, field_name in enumerate(remaining):
            if i == len(remaining) - 1:
                target_conf[field_name] = value
            else:
                target_conf.setdefault(field_name, {})
                target_conf = target_conf[field_name]

    def parse_pairs_key_value(pairs: List[Tuple[str, str]]):
        conf = {}
        for key, value_str in pairs:
            parse_single_key_value(conf, key, value_str)
        return conf

    revise_conf = parse_pairs_key_value(revise)
    add_conf = parse_pairs_key_value(add)

    return revise_conf, add_conf


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def initialize():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--print", action="store_true", help="Show the default config"
    )
    parser.add_argument(
        "-c",
        "--conf",
        help="Provide a new config. The default config will be ignored.",
    )
    args, remained = parser.parse_known_args()

    current_script = sys.argv[0]
    package_path = Path(__package__)
    default_conf_path = (
        package_path.parent / "confs" / f"{Path(current_script).stem}.yaml"
    )
    assert (
        args.conf is not None
    ) or default_conf_path.is_file(), (
        f"Either --conf or {default_conf_path} must be given."
    )
    conf_path = args.conf or default_conf_path

    OmegaConf.register_new_resolver("eval", eval)
    with open(conf_path) as f:
        conf = OmegaConf.load(f)

    if args.print:
        conf = OmegaConf.to_container(conf, resolve=True)
        print(yaml.dump(conf))
        exit(0)

    revise_conf, add_conf = parse_overrides(remained)

    conf = OmegaConf.create(conf)
    OmegaConf.set_struct(conf, True)
    conf = OmegaConf.merge(conf, revise_conf)
    with open_dict(conf):
        conf = OmegaConf.merge(conf, add_conf)
    conf = OmegaConf.to_container(conf, resolve=True, throw_on_missing=True)
    logger.info(f"Config:\n{yaml.safe_dump(conf)}")

    if "expdir" in conf:
        expdir = Path(conf["expdir"])
        expdir.mkdir(exist_ok=True, parents=True)

        conf_dir = expdir / "confs"
        conf_dir.mkdir(exist_ok=True, parents=True)
        with open(conf_dir / f"{get_time_tag()}.yaml", "w") as f:
            yaml.safe_dump(conf, f)

    seed = conf.get("seed", 0)
    seed_all(seed)

    return conf


def get_tester(
    expdir: str,
    trainer_conf: dict,
):
    tester = pl.Trainer(default_root_dir=expdir, **trainer_conf)
    return tester


def get_trainer(
    expdir: str,
    trainer_conf: dict,
    save_steps: int,
    valid_steps: int,
    valid_metric: str,
    valid_higher_better: bool,
):
    last_checkpointing = ModelCheckpoint(
        dirpath=expdir,
        monitor="global_step",
        mode="max",
        filename="global_step-{global_step:.0f}",
        save_top_k=2,
        every_n_train_steps=save_steps,
        auto_insert_metric_name=False,
    )
    valid_checkpointing = ModelCheckpoint(
        dirpath=expdir,
        monitor=valid_metric,
        mode="max" if valid_higher_better else "min",
        filename=f"best_{valid_metric.replace('/', '_')}-"
        + "{"
        + valid_metric
        + ":.2f}",
        save_top_k=1,
        every_n_train_steps=valid_steps,
        auto_insert_metric_name=False,
    )
    tb_logger = TensorBoardLogger(
        save_dir=expdir,
        name=None,
        version="tb",
    )
    trainer = pl.Trainer(
        logger=tb_logger,
        callbacks=[valid_checkpointing, last_checkpointing],
        **trainer_conf,
    )
    return trainer


_history = defaultdict(list)


class benchmark(ContextDecorator):
    def __init__(self, name: str, freq: int = 1) -> None:
        super().__init__()
        self.name = name
        self.freq = freq

    def __enter__(self):
        torch.cuda.synchronize()
        self.start = time()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        torch.cuda.synchronize()
        seconds = time() - self.start

        global _history
        _history[self.name].append(seconds)
        if len(_history[self.name]) % self.freq == 0:
            logger.warning(
                f"{self.name}: {seconds} secs, avg {np.array(_history[self.name]).mean()} secs"
            )
