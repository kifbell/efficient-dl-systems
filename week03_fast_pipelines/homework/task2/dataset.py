from typing import Optional
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

import random
import torch
import torchtext
from torch.utils.data.dataset import Dataset
from torch.utils.data import Sampler

from datasets import load_dataset
from torchtext.vocab import build_vocab_from_iterator

MAX_LENGTH = 640
N_TOKENS = 5000

DATASET_DIRECTORY = Path("task2_dataset/")
DATASET_DIRECTORY.mkdir(exist_ok=True)

VOCAB_PATH = DATASET_DIRECTORY / "vocab.pt"
DATASET_PATH = DATASET_DIRECTORY / "dataset.pt"
DEBUG_DATASET_SIZE = 10000
DEBUG_DATASET_PATH = DATASET_DIRECTORY / f"dataset_first_{DEBUG_DATASET_SIZE}.pt"


def create_dataset(ntoken: int = N_TOKENS, force_recompute: bool = False):
    if DATASET_PATH.exists() and not force_recompute:
        return

    print("Download dataset")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", ignore_verifications=True)["text"]

    print("Tokenize sentences")
    tokenizer = torchtext.data.utils.get_tokenizer("basic_english")
    dataset = [tokenizer(x) for x in tqdm(dataset)]

    print("Build vocabulary")
    vocab = build_vocab_from_iterator(tqdm(dataset), max_tokens=ntoken, specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    print("Map sentences into tokens")
    dataset = [torch.IntTensor(vocab(x)) for x in tqdm(dataset)]
    dataset = [x for x in tqdm(dataset) if len(x) > 0]

    print("Save dataset")
    torch.save(vocab, VOCAB_PATH)
    print("Save vocabulary")
    torch.save(dataset, DATASET_PATH)
    print("Save debug dataset")
    torch.save(dataset[:DEBUG_DATASET_SIZE], DEBUG_DATASET_PATH)

    print("Create dataset: Success")


def load_custom_dataset(data_path: Path) -> list[torch.Tensor]:
    print(f"Load dataset from {data_path}")
    dataset = torch.load(data_path)
    print("Load dataset: success")
    return dataset


class BrainDataset(Dataset):
    def __init__(self, data_path: Path = DATASET_PATH, max_length: int = MAX_LENGTH):
        self.dataset = load_custom_dataset(data_path)
        self.max_length = max_length

    def __getitem__(self, idx: int):
        s = self.dataset[idx]
        new_length = min(len(s), self.max_length)

        mask = torch.zeros(self.max_length, dtype=torch.bool)
        mask[:new_length] = 1

        padded_s = torch.zeros(self.max_length, dtype=s.dtype)
        padded_s[:new_length] = s[:new_length]

        return padded_s, mask

    def __len__(self):
        return len(self.dataset)


class BigBrainDataset(Dataset):
    def __init__(self, data_path: Path = DATASET_PATH, max_length: int = MAX_LENGTH):
        self.dataset = load_custom_dataset(data_path)
        self.max_length = max_length

    def __getitem__(self, idx: int):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)


class UltraBigBrainDataset(Dataset):
    def __init__(self, data_path: Path = DATASET_PATH, max_length: int = MAX_LENGTH, n_bins: int = 1):
        self.dataset = load_custom_dataset(data_path)
        self.max_length = max_length

    def __getitem__(self, idx: int):
        return self.dataset[idx][: self.max_length]

    def __len__(self):
        return len(self.dataset)


def collate_fn(batch: list[torch.Tensor], max_length: Optional[int] = MAX_LENGTH) -> tuple[torch.Tensor, torch.Tensor]:
    # Clip to maximum length
    batch = [b[:max_length] for b in batch]
    # Calculate length of output batch
    result_length = max([len(b) for b in batch])
    # Construct new batch with mask
    result = torch.zeros((len(batch), result_length), dtype=batch[0].dtype)
    mask = torch.zeros_like(result, dtype=torch.bool)
    for i, tensor in enumerate(batch):
        result[i, : tensor.size(0)] = tensor
        mask[i, : tensor.size(0)] = 1
    return result, mask


class UltraBigBrainBatchSampler(Sampler):
    def __init__(self, dataset: UltraBigBrainDataset, k: int, batch_size: int, max_length: Optional[int] = MAX_LENGTH):
        lengths = [len(x) for x in dataset]
        groups = defaultdict(list)

        # group by lengths into buckets
        for i, l in enumerate(lengths):
            l = min(l, max_length)
            l -= 1
            assert l >= 0
            groups[l // k].append(i)
        print(f"{len(groups)} groups:")
        for g, indices in groups.items():
            print(f"[{g * k}, {g * k + k}):\t{len(indices)}")

        # construct batch indices
        self.batches = []
        for g, indices in groups.items():
            # shuffle indices
            random.shuffle(indices)
            for b_start_ind in range(0, len(indices), batch_size):
                self.batches.append(indices[b_start_ind : b_start_ind + batch_size])

        # shuffle batches
        random.shuffle(self.batches)
        assert sum((len(b) for b in self.batches)) == len(dataset)

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)
