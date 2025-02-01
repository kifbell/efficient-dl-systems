import torch
from pathlib import Path

from modeling.diffusion import DiffusionModel
from config import DiffusionConfig, OptimizerConfig
from torch.optim import Optimizer


def save_model(model: DiffusionModel, optimizer: Optimizer, save_directory: Path, epoch: int):
    save_directory.mkdir(exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        save_directory / f"model_{epoch}.pt",
    )


def load_model(
    model_cfg: DiffusionConfig,
    optimizer_cfg: OptimizerConfig | None,
    save_directory: str,
    device: str,
    epoch: int | None = None,
):

    assert Path(save_directory).exists()
    if epoch is None:
        epochs = [int(x.name[x.name.rfind("_") + 1 : x.name.rfind(".")]) for x in Path(save_directory).iterdir()]
        epochs.sort()
        epoch = epochs[-1]

    model = model_cfg.instantiate()
    model.to(device)
    checkpoint = torch.load(Path(save_directory) / f"model_{epoch}.pt")
    model.load_state_dict(checkpoint["model"])

    if optimizer_cfg is not None:
        optimizer = optimizer_cfg.instantiate(params=model.parameters())
        optimizer.load_state_dict(checkpoint["optimizer"])
        return model, optimizer
    return model
