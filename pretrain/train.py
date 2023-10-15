"""
Trainer for model tuned from scratch.
"""

from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import numpy as np

from model import Transformer, TransformerConfig

print("Model imported!")
device = "mps"

learning_rate = 6e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
batch_size = 32
accumulation_steps = 4
epochs = 20

config = TransformerConfig(
    block_size=64,
    embedding_size=512,
    num_hidden_layers=8,
    num_attention_heads=8,
    dropout=0.1,
    bias=False,
)


class LyricsDataset(Dataset):
    """Dataset for lyrics."""

    def __init__(self, mode, block_size=1024):
        self.mode = mode
        if mode not in ["train", "validation", "inference"]:
            raise ValueError("Expected train, validation or inference as mode.")
        if self.mode == "train":
            self.data = np.memmap("spinetta_ai/src/data/pretrain_train.bin", dtype=np.uint16, mode="r")
        elif mode == "validation":
            self.data = np.memmap("spinetta_ai/src/data/pretrain_val.bin", dtype=np.uint16, mode="r")
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, index):
        X = torch.tensor(self.data[index : index + self.block_size].astype(np.int64))
        if self.mode in ["train", "validation"]:
            y = torch.tensor(self.data[index + 1 : index + self.block_size + 1].astype(np.int64))

            return X, y
        return X


if __name__ == "__main__":
    model = Transformer(config)
    model = model.to(device)

    train_dataset = LyricsDataset(mode="train", block_size=config.block_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    eval_dataset = LyricsDataset(mode="validation", block_size=config.block_size)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(beta1, beta2),
    )

    train_losses = []
    eval_losses = []

    for epoch in range(epochs):
        train_loss = 0
        model.train()

        for i, (X, y) in enumerate(train_loader):
            X = X.to(device)
            y = y.to(device)

            outputs = model(X)
            loss = criterion(outputs.view(-1, outputs.size(-1)), y.view(-1)) / accumulation_steps

            loss.backward()
            train_loss += loss.item()

            if grad_clip > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            if i % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if i % 500 == 0:
                print(f"Epoch {epoch} | Iter {i} | Training Loss {loss.item()}")

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        eval_loss = 0

        with torch.no_grad():
            for X_eval, y_eval in eval_loader:
                X_eval = X_eval.to(device)
                y_eval = y.to(device)

                eval_outputs = model(X_eval)
                eval_loss += criterion(eval_outputs.view(-1, eval_outputs.size(-1)), y_eval.view(-1)).item()

        eval_loss /= len(eval_loader)
        eval_losses.append(eval_loss)

        print(f"Epoch {epoch} | Evaluation Loss: {eval_loss}")

        # Save model checkpoint based on evaluation loss
        torch.save(model.state_dict(), "model.ckpt")
        print("Model checkpoint saved!")
