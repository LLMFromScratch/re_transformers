from typing import Tuple

import torch
import torch.nn as nn
from transformers import Qwen3Config
from transformers.activations import ACT2FN
from transformers.integrations import use_kernel_forward_from_hub


@use_kernel_forward_from_hub("RMSNorm")
class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()

        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * \
            torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen3MLP(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()

        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., :x.size(-1) // 2]
    x2 = x[..., x.size(-1) // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotaty_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


# Questions
# 1. `Qwen3Config::keys_to_ignore_at_inference`
# 2. `Qwen3Config::base_model_pp_plan`
