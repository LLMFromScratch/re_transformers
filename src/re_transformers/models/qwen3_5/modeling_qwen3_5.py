import torch
import torch.nn as nn


class Qwen3_5VisionRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()

        inv_freq = 1.0 / \
            (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, dtype=self.inv_freq.dtype,
                           device=self.inv_freq.device)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs
