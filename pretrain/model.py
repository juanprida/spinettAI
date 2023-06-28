"""
Transformer model for pre-training. Decoder only.
Adaptation from Karpathy's nanoGPT
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    embedding_size: int = 768
    dropout: float = 0.1

class CausalSelfAttention(nn.Module):
    """Causal self-attention layer."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_k = config.embedding_size // config.num_attention_heads
        self.query = nn.Linear(config.embedding_size, self.d_k)
        self.key = nn.Linear(config.embedding_size, self.d_k)
        self.value = nn.Linear(config.embedding_size, self.d_k)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.projection_dropout = nn.Dropout(config.dropout)
        self.flash = hasattr(
            torch.nn.functional, "scaled_dot_product_attention"
        )
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
            self.mask = torch.tril(
                torch.ones(config.block_size, config.block_size)
            )

    def forward(self, x):
        # (batch_size, seq_len, embedding_size // num_attention_heads)
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        if self.flash:
            # (seq_len, seq_len)
            attention = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=None
            )
        else:
            # (seq_len, seq_len)
            attention = query @ key.transpose(-2, -1) / (self.d_k**0.5)
            attention = attention.masked_fill(self.mask, -float("inf"))
            attention = torch.softmax(attention, dim=-1)
            attention = self.attention_dropout(attention)
        # (batch_size, seq_len, embedding_size // num_attention_heads)
        projection = self.projection_dropout(attention @ value)
        return projection


class MultiHeadAttention(nn.Module):
    """Multi-head attention layer."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attention = nn.ModuleList(
            [
                CausalSelfAttention(config)
                for _ in range(config.num_attention_heads)
            ]
        )
        self.projection = nn.Linear(
            config.embedding_size, config.embedding_size
        )

    def forward(self, x):
        # (batch_size, seq_len, embedding_size)
        x = torch.cat(
            [
                self.attention[i](x)
                for i in range(self.config.num_attention_heads)
            ],
            dim=-1,
        )
        # (batch_size, seq_len, embedding_size)
        return self.projection(x)


class MLP(nn.Module):
    """MLP layer."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.embedding_size, 4 * config.embedding_size)
        self.fc2 = nn.Linear(4 * config.embedding_size, config.embedding_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # (batch_size, seq_len, 4 * embedding_size)
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)

        # (batch_size, seq_len, embedding_size)
        return self.fc2(x)


class Block(nn.Module):
    """Block of Transformer."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attention = MultiHeadAttention(config)
        self.layernorm1 = nn.LayerNorm(config.embedding_size)
        self.mlp = MLP(config)
        self.layernorm2 = nn.LayerNorm(config.embedding_size)

    def forward(self, x):
        # (batch_size, seq_len, embedding_size)

        x = self.layernorm1(x + self.attention(x))

        # (batch_size, seq_len, embedding_size)
        x = self.layernorm2(x + self.mlp(x))

        # (batch_size, seq_len, embedding_size)
        return x


class Transformer(nn.Module):
    """Transformer model for pre-training. Decoder only."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.embedding_size),
                "wpe": nn.Embedding(config.block_size, config.embedding_size),
                "drop": nn.Dropout(config.dropout),
                "block": nn.ModuleList(
                    [Block(config) for _ in range(config.num_hidden_layers)]
                ),
                "ln_f": nn.LayerNorm(config.embedding_size),
            }
        )
        self.lm_head = nn.Linear(
            config.embedding_size, config.vocab_size, bias=False
        )
        self.model.wte.weight = self.lm_head.weight

    def forward(self, x):
        device = x.device
        pos = torch.arange(0, x.size(1), dtype=torch.long, device=device)

        # (batch_size, seq_len, embedding_size)
        tok_emb = self.model["wte"](x)
        pos_emb = self.model["wpe"](pos)

        x = tok_emb + pos_emb
        x = self.model["drop"](x)
        for block in self.model["block"]:
            x = block(x)

        x = self.model["ln_f"](x)
        logits = self.lm_head(x)

        return logits
