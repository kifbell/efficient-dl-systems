#!/usr/bin/env twix-python
import hydra
from omegaconf import DictConfig
from config import CIFAR10Config


@hydra.main(config_path="configs/dataset", config_name="cifar10.yaml", version_base=None)
def prepare_dataset(cfg: DictConfig):
    CIFAR10Config.from_omegaconf(cfg).instantiate()
    print("downloaded dataset")


if __name__ == "__main__":
    prepare_dataset()
