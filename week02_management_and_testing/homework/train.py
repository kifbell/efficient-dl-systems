#!/usr/bin/env twix-python
import torch
import hydra

import wandb
import random
import numpy as np

from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision import transforms
from modeling.training import generate_samples, train_epoch
from modeling.save_utils import save_model
from config import Config
from omegaconf import OmegaConf


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# @hydra.main(config_path="configs", config_name="config.yaml", version_base=None)
def train():
    cfg = OmegaConf.create(OmegaConf.load("params.yaml"))
    assert isinstance(cfg, DictConfig)

    config = Config.from_omegaconf(cfg, resolve=True)
    seed_all(1)
    print(config.model_dump_json(indent=4))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    wandb_cfg = config.train.wandb
    wandb.init(
        project=wandb_cfg.project,
        name=wandb_cfg.run_name,
        config=config.model_dump(),
    )

    ddpm = config.model.instantiate(
        sample_transform=transforms.Normalize((-1, -1, -1), (2, 2, 2)), eps_model=config.model.eps_model.instantiate()
    )
    ddpm.to(device)
    train_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        + [a.instantiate() for a in config.train.augmentations]
    )

    dataset = config.dataset.instantiate(transform=train_transforms)
    dataloader = DataLoader(
        dataset, batch_size=config.train.loop.batch_size, num_workers=config.train.loop.num_workers, shuffle=True
    )

    optimizer = config.train.optimizer.instantiate(params=ddpm.parameters())

    save_samples_directory = config.train.save.samples_directory
    save_samples_directory.mkdir(exist_ok=True)

    for epoch in range(1, config.train.loop.num_epochs + 1):
        train_epoch(ddpm, dataloader, optimizer, device, log=True)
        if epoch % config.train.save.samples_frequency_epochs == 0 or epoch == config.train.loop.num_epochs:
            save_model(ddpm, optimizer, config.train.save.model_directory, epoch)

        generate_samples(ddpm, device, save_samples_directory / f"{epoch:02d}.png", log=True)
    print("finished training")


if __name__ == "__main__":
    train()
