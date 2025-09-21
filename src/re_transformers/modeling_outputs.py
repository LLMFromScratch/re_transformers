from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from transformers.utils import ModelOutput


@dataclass
class BaseModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None  # type: ignore
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
