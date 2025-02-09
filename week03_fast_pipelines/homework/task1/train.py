import torch
import wandb
import argparse
import numpy as np
from torch import nn
from tqdm.auto import tqdm
from math import sqrt

from unet import Unet
from dataset import get_train_data


class StaticScaler:
    def __init__(self, scale: float) -> None:
        self._scale = scale
        self._unscale_called = False
        self._contains_nans = False

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss * self._scale

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        assert not self._unscale_called
        assert not self._contains_nans
        self._unscale_called = True
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad /= self._scale
                    self._contains_nans = not param.grad.isfinite().all().item()
                    if self._contains_nans:
                        return

    def step(self, optimizer: torch.optim.Optimizer) -> bool:
        if not self._unscale_called:
            self.unscale_(optimizer)
        if not self._contains_nans:
            optimizer.step()
            step_called = True
        else:
            step_called = False
        self._unscale_called = False
        self._contains_nans = False
        return step_called

    def update(self) -> None:
        pass


class DynamicScaler(StaticScaler):
    def __init__(self, scale: float, growth_factor: float, backoff_factor: float, growth_interval: int) -> None:
        super().__init__(scale)
        self._n_skipped = 0
        self._backoff_factor = backoff_factor
        self._growth_factor = growth_factor
        self._growth_interval = growth_interval

    def step(self, optimizer: torch.optim.Optimizer) -> bool:
        if not (result := super().step(optimizer)):
            self._n_skipped += 1
        else:
            self._n_skipped = 0
        return result

    def update(self):
        if self._n_skipped >= self._growth_interval:
            self._scale *= self._growth_factor
        elif self._n_skipped > 0:
            self._scale *= self._backoff_factor


def train_epoch(
    train_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    scaler: StaticScaler | DynamicScaler,
    device: torch.device,
    log: bool,
) -> None:
    model.train()

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    n_correct = 0
    n_examples = 0
    for _, (images, labels) in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        # Calculate original gradient norm
        grad_norm = (
            sqrt(sum([(torch.norm(p.grad) ** 2).sum().item() for p in model.parameters()]))
            if not scaler._contains_nans
            else np.nan
        )

        scaler.step(optimizer)
        scaler.update()

        new_n_correct = ((outputs.detach() > 0.5) == labels).float().sum().item()
        new_n_examples = torch.numel(labels)

        n_correct += new_n_correct
        n_examples += new_n_examples

        batch_accuracy = new_n_correct / new_n_examples * 100
        loss = loss.item()
        if log:
            wandb.log(
                {"loss": loss, "batch_accuracy_%": batch_accuracy, "grad_norm": grad_norm, "scale": scaler._scale}
            )

        pbar.set_description(
            f"Loss: {round(loss, 5)}; Accuracy: {round(batch_accuracy, 4)}%; Norm: {grad_norm}; Scale: {scaler._scale}"
        )
    total_accuracy = n_correct / n_examples * 100
    wandb.log({"total_accuracy_%": total_accuracy})


from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
import hydra
from omegaconf import DictConfig


@dataclass
class ScalerConfig:
    scaler_type: str = field(default="static")
    scale: int = field(default=2**16)
    growth_factor: float = field(default=2.0)
    backoff_factor: float = field(default=0.5)
    growth_interval: float = field(default=16.0)


@dataclass
class TrainConfig:
    scaler: ScalerConfig = field(default_factory=ScalerConfig)
    wandb_name: str = field(default="")
    batch_size: int = field(default=128)
    num_workers: int = field(default=2)
    num_epochs: int = field(default=5)


cs = ConfigStore.instance()
cs.store(name="default", node=TrainConfig)
cs.store(name="default", node=ScalerConfig, group="scaler")


@hydra.main(config_path="configs", config_name="default")
def train(cfg: DictConfig) -> None:
    print(f"Config: {cfg}")
    assert cfg.scaler.scaler_type in ["static", "dynamic"], cfg.scaler.scaler_type
    assert cfg.scaler.scale > 0, cfg.scaler.scale

    log = cfg.wandb_name != ""

    scaler_by_type = {
        "static": StaticScaler(cfg.scaler.scale),
        "dynamic": DynamicScaler(
            cfg.scaler.scale,
            growth_factor=cfg.scaler.growth_factor,
            backoff_factor=cfg.scaler.backoff_factor,
            growth_interval=cfg.scaler.growth_interval,
        ),
    }
    scaler = scaler_by_type[cfg.scaler.scaler_type]

    device = torch.device("cuda:0")
    model = Unet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_loader = get_train_data(batch_size=cfg.batch_size, num_workers=cfg.num_workers)

    if log:
        wandb.init(project="edl-hw2-task-1", name=cfg.wandb_name)
        wandb.config = dict(cfg)

    for _ in range(0, cfg.num_epochs):
        train_epoch(train_loader, model, criterion, optimizer, scaler, device=device, log=log)


if __name__ == "__main__":
    train()
