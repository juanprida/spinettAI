"""Train transformer model. """

from dataclasses import dataclass
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np

from pretrain.model import TransformerConfig, Transformer


@dataclass
class TrainingConfig:
    """Configuration for `train`."""

    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    adam_epsilon: float = 1e-8
    warmup_steps: int = 10000
    gradient_accumulation_steps: int = 1
    seed: int = 42
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"


class LyricsDataset(Dataset):
    """Dataset for lyrics."""

    def __init__(self, mode, block_size=1024):
        dir_location = os.path.split(
            os.path.dirname(os.path.abspath("__file__"))
        )[0]
        self.mode = mode
        if mode not in ["train", "validation", "inference"]:
            raise ValueError("Expected train, validation or inference as mode.")
        if self.mode == "train":
            self.data = np.memmap(
                # os.path.join(dir_location, "data/pretrain_train.bin"),
                "data/pretrain_train.bin",
                dtype=np.uint16,
                mode="r"
            )
        elif mode == "validation":
            self.data = np.memmap(
                # os.path.join(dir_location, "data/pretrain_val.bin"),
                "data/pretrain_val.bin",
                dtype=np.uint16,
                mode="r"
            )
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, index):
        X = torch.tensor(
            self.data[index : index + self.block_size].astype(np.int64)
        )
        if self.mode in ["train", "validation"]:
            y = torch.tensor(
                self.data[index + 1 : index + self.block_size + 1].astype(
                    np.int64
                )
            )

            return X, y
        return X
