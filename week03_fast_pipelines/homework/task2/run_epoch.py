import torch
import wandb
from enum import Enum
from tqdm import tqdm
from torch import nn
from time import time
from pathlib import Path
from torch.utils.data import DataLoader


from transformer import PositionalEncoding
from dataset import (
    BrainDataset,
    BigBrainDataset,
    UltraDuperBigBrainDataset,
    UltraDuperBigBrainBatchSampler,
    create_dataset,
    collate_fn,
)
from dataset import N_TOKENS, DATASET_PATH, DEBUG_DATASET_PATH


class DataMode(Enum):
    BRAIN = 1
    BIG_BRAIN = 2
    ULTRA_DUPER_BIG_BRAIN = 3


class GPT2Model(nn.Module):
    def __init__(self, ntoken: int, d_model: int = 1024, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Embedding(num_embeddings=ntoken, embedding_dim=d_model),
            PositionalEncoding(d_model=d_model, dropout=dropout),
        )
        self.decoder = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)

    def forward(self, src, src_mask):
        src = self.encoder(src)
        return self.decoder(src, src, tgt_key_padding_mask=src_mask, memory_key_padding_mask=src_mask)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_size_mb(model):
    total_params = count_parameters(model)
    # Assuming each parameter is a 32-bit float (4 bytes)
    size_bytes = total_params * 4
    size_mb = size_bytes / (1024 * 1024)
    return size_mb


def get_gpt2_model(ntoken: int) -> torch.nn.Module:
    model = GPT2Model(ntoken=ntoken)
    n_parameters = count_parameters(model)
    n_parameters = f"{n_parameters:_}".replace("_", " ")
    print(f"Number of parameters: {n_parameters}")
    print(f"Model size: {calculate_size_mb(model):.1f} MB")
    # print(model)
    return model


class Timer:
    def __init__(self) -> None:
        self.start_time = None

    def start_timer(self):
        torch.cuda.synchronize()
        self.start_time = time()

    def end_timer(self) -> float:
        torch.cuda.synchronize()
        result = time() - self.start_time
        return result


def run_epoch(
    data_mode: DataMode, data_path: str = DATASET_PATH, batch_size=32, device="cuda:0", k: int = None
) -> None:
    create_dataset(ntoken=N_TOKENS)
    device = torch.device(device)

    model = get_gpt2_model(ntoken=N_TOKENS).to(device)

    collate_fn = None
    batch_sampler = None
    if data_mode is DataMode.BRAIN:
        name = "Brain"
        dataset = BrainDataset(data_path)
    elif data_mode is DataMode.BIG_BRAIN:
        name = "BigBrain"
        dataset = BigBrainDataset(data_path)
        collate_fn = collate_fn
    elif data_mode is DataMode.ULTRA_DUPER_BIG_BRAIN:
        assert k is not None
        name = f"UltraDuperBigBrain_k={k}"
        dataset = UltraDuperBigBrainDataset(data_path)
        collate_fn = collate_fn
        batch_sampler = UltraDuperBigBrainBatchSampler(dataset, batch_size=batch_size, k=k)
        batch_size = 1
    else:
        assert False, "Unreachable"

    with wandb.init(project="efdl-hw-2-task-2", name=name) as run:
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, batch_sampler=batch_sampler)
        timer = Timer()

        print(f"Start train loop for {name}")
        for src, src_mask in tqdm(dataloader):
            src, src_mask = src.to(device), src_mask.to(device)
            timer.start_timer()
            result = model(src, src_mask)
            time = timer.end_timer()
            wandb.log({f"time": time, "batch_length": src.size(1)})

        print("Train loop: success")


def main():
    data_path = DATASET_PATH
    for k in [1, 5, 10, 20, 50, 640]:
        run_epoch(data_mode=DataMode.ULTRA_DUPER_BIG_BRAIN, data_path=data_path, k=k)
    run_epoch(data_mode=DataMode.BIG_BRAIN, data_path=data_path)
    run_epoch(data_mode=DataMode.BRAIN, data_path=data_path)


if __name__ == "__main__":
    main()
