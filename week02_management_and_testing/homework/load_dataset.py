#!/usr/bin/env twix-python
import hydra
from omegaconf import DictConfig
from config import CIFAR10Config


from omegaconf import OmegaConf


def prepare_dataset():
    cfg = OmegaConf.create(OmegaConf.load("params.yaml"))
    assert isinstance(cfg, DictConfig)
    CIFAR10Config.from_omegaconf(cfg.dataset).instantiate()
    print("downloaded dataset")


if __name__ == "__main__":
    prepare_dataset()
