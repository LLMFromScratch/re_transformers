from typing import Optional, Tuple

from transformers import PretrainedConfig
from transformers.utils import is_torch_available
from transformers.modeling_rope_utils import (
    _compute_dynamic_ntk_parameters,
    _compute_linear_scaling_rope_parameters,
    _compute_llama3_parameters,
    _compute_longrope_parameters,
    _compute_yarn_parameters,
)


if is_torch_available():
    import torch


def _compute_default_rope_parameters(
    config: Optional[PretrainedConfig] = None,
    device: Optional["torch.device"] = None,
    seq_len: Optional[int] = None
) -> Tuple["torch.Tensor", float]:
    base = config.rope_theta
    head_dim = getattr(
        config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    dim = int(head_dim * partial_rotary_factor)

    inv_freq = 1 / \
        (base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float) / dim))
    attention_factor = 1.0
    return inv_freq, attention_factor


ROPE_INIT_FUNCTIONS = {
    "default": _compute_default_rope_parameters,
    "linear": _compute_linear_scaling_rope_parameters,
    "dynamic": _compute_dynamic_ntk_parameters,
    "yarn": _compute_yarn_parameters,
    "longrope": _compute_longrope_parameters,
    "llama3": _compute_llama3_parameters,
}
