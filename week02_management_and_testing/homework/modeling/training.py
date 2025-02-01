import torch
import wandb
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from pathlib import Path

from modeling.diffusion import DiffusionModel


def train_step(model: DiffusionModel, inputs: torch.Tensor, optimizer: Optimizer, device: str):
    optimizer.zero_grad()
    inputs = inputs.to(device)
    loss = model(inputs)
    loss.backward()
    optimizer.step()
    return loss.item()


def train_epoch(model: DiffusionModel, dataloader: DataLoader, optimizer: Optimizer, device: str, log: bool = False):
    model.train()
    pbar = tqdm(dataloader)
    loss_ema = None
    if log:
        wandb.log({"learning_rate": next(iter(optimizer.param_groups))["lr"]})
    for i, (x, _) in enumerate(pbar):
        if i == 0 and log:
            wandb.log(
                {"input": wandb.Image(make_grid(x, nrow=16).permute(1, 2, 0).numpy(), caption="Input's first batch")}
            )
        train_loss = train_step(model, x, optimizer, device)
        loss_ema = train_loss if loss_ema is None else 0.9 * loss_ema + 0.1 * train_loss
        pbar.set_description(f"loss: {loss_ema:.4f}")
        if log:
            wandb.log({"train_loss": train_loss, "train_loss_ema": loss_ema})


def generate_samples(model: DiffusionModel, device: str, path: Path, log: bool = False):
    model.eval()
    with torch.no_grad():
        samples = model.sample(8, (3, 32, 32), device=device)
        grid = make_grid(samples, nrow=4)
        save_image(grid, path)
        if log:
            image = wandb.Image(grid.cpu().detach().permute(1, 2, 0).numpy(), caption="Generated images")
            wandb.log({"samples": image})
    return grid
