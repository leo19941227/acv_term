import logging
from typing import List

import torch
from torch.utils.data.dataloader import DataLoader
from acv_term.utils import initialize, get_trainer, find_last_ckpt, merge_list_of_dict


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
