"""
Transformer model for pre-training. Decoder only.
"""
from dataclasses import dataclass

from transformers import GPT2Tokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    """Configuration for `Transformer`."""

    block_size: int = 1024
    vocab_size: int = 50304
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    embedding_size: int = 768
    dropout: float = 0.1
    bias: bool = False


class CausalSelfAttention(nn.Module):
    """Causal self-attention layer."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_k = config.embedding_size // config.num_attention_heads
        self.query = nn.Linear(
            config.embedding_size, self.d_k, bias=config.bias
        )
        self.key = nn.Linear(config.embedding_size, self.d_k, bias=config.bias)
        self.value = nn.Linear(
            config.embedding_size, self.d_k, bias=config.bias
        )
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, block_size, embedding_size // num_attention_heads)
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        if self.flash:
            # (block_size, block_size)
            return torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=None
            )

        else:
            # (block_size, block_size)
            attention = query @ key.transpose(-2, -1) / (self.d_k**0.5)
            attention = attention.masked_fill(self.mask, -float("inf"))
            attention = torch.softmax(attention, dim=-1)
            attention = self.attention_dropout(attention)
            return self.projection_dropout(attention @ value)


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
            config.embedding_size, config.embedding_size, bias=config.bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, block_size, embedding_size)
        x = torch.cat(
            [
                self.attention[i](x)
                for i in range(self.config.num_attention_heads)
            ],
            dim=-1,
        )
        # (batch_size, block_size, embedding_size)
        return self.projection(x)


class MLP(nn.Module):
    """MLP layer."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(
            config.embedding_size, 4 * config.embedding_size, bias=config.bias
        )
        self.fc2 = nn.Linear(
            4 * config.embedding_size, config.embedding_size, bias=config.bias
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, block_size, 4 * embedding_size)
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)

        # (batch_size, block_size, embedding_size)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, block_size, embedding_size)

        x = self.attention(self.layernorm1(x)) + x

        # (batch_size, block_size, embedding_size)
        x = self.mlp(self.layernorm2(x)) + x

        # (batch_size, block_size, embedding_size)
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
            config.embedding_size, config.vocab_size, bias=config.bias
        )

        # report number of parameters - taken from nano-gpt repo
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """Return the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.model.wpe.weight.numel()
        return n_params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        pos = torch.arange(0, x.size(1), dtype=torch.long, device=device)
        # (batch_size, block_size, embedding_size)
        tok_emb = self.model["wte"](x)
        pos_emb = self.model["wpe"](pos)

        x = tok_emb + pos_emb
        x = self.model["drop"](x)
        for block in self.model["block"]:
            x = block(x)

        x = self.model["ln_f"](x)
        logits = self.lm_head(x)

        return logits

    def generate(
        self,
        tokenizer: GPT2Tokenizer,
        prompt: str,
        max_len: int = 100,
        temperature: float = 1.0,
        device: torch.device = torch.device("mps"),
    ) -> str:
        """
        Generate lyrics from a prompt.
        """
        # Tokenize the prompt
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
        prompt_tokens = torch.tensor(
            prompt_tokens, dtype=torch.long, device=device
        )
        prompt_tokens = prompt_tokens.unsqueeze(0)

        # Generate lyrics
        self.eval()
        with torch.no_grad():
            for _ in range(max_len):
                logits = (
                    self(prompt_tokens[:, -self.config.block_size :])
                    / temperature
                )
                logits = logits[0, -1, :]
                probs = F.softmax(logits, dim=-1)
                next_token_id = torch.multinomial(
                    probs, num_samples=1
                ).unsqueeze(0)
                if next_token_id == tokenizer.eos_token_id:
                    break
                prompt_tokens = torch.cat(
                    (prompt_tokens, next_token_id), dim=1
                )

        # Decode the generated lyrics
        generated = tokenizer.decode(
            prompt_tokens.squeeze().tolist(), clean_up_tokenization_spaces=True
        )
        return generated
