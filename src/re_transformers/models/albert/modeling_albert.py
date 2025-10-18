from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from transformers.pytorch_utils import apply_chunking_to_forward

from .configuration_albert import AlbertConfig
from ...modeling_outputs import BaseModelOutput


ALBERT_ATTENTION_CLASSES = {
    "eager": AlbertAttention,
    "sdpa": AlbertSdpaAttention,
}


class AlbertLayer(nn.Module):
    def __init__(self, config: AlbertConfig) -> None:
        super().__init__()

        self.config = config
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1

        self.attention = ALBERT_ATTENTION_CLASSES[config._attn_implementation](config)  #type: ignore
        self.ffn = nn.Linear(config.hidden_size, config.intermediate_size)
        self.ffn_output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation = ACT2FN[config.hidden_act]
        self.full_layer_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attention_output = self.attention(hidden_states, attention_mask, head_mask, output_attentions)
        ffn_output = apply_chunking_to_forward(
            self.ff_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output[0],
        )
        hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
        return (hidden_states, ) + attention_output[1:]

    def ff_chunk(self, attention_output: torch.Tensor) -> torch.Tensor:
        ffn_output = self.ffn(attention_output)
        ffn_output = self.activation(ffn_output)
        ffn_output = self.ffn_output(ffn_output)
        return ffn_output


class AlbertLayerGroup(nn.Module):
    def __init__(self, config: AlbertConfig) -> None:
        super().__init__()

        self.albert_layers = nn.ModuleList([
            AlbertLayer(config) for _ in range(config.inner_group_num)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        layer_hidden_states = ()
        layer_attentions = ()

        for layer_index, albert_layer in enumerate(self.albert_layers):
            layer_output = albert_layer(
                hidden_states,
                attention_mask,
                head_mask[layer_index],  # type: ignore
                output_attentions,
            )
            hidden_states = layer_output[0]

            if output_attentions:
                layer_attentions = layer_attentions + (layer_output[1], )
            if output_hidden_states:
                layer_hidden_states = layer_hidden_states + (hidden_states, )

        outputs = (hidden_states, )
        if output_hidden_states:
            outputs = outputs + (layer_hidden_states, )
        if output_attentions:
            outputs = outputs + (layer_attentions, )
        return outputs  # type: ignore


class AlbertTransformer(nn.Module):
    def __init__(self, config: AlbertConfig) -> None:
        super().__init__()

        self.config = config
        self.embedding_hidden_mapping_in = nn.Linear(
            config.embedding_size,
            config.hidden_size,
        )
        self.albert_layer_groups = nn.ModuleList([
            AlbertLayerGroup(config) for _ in range(config.num_hidden_groups)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[BaseModelOutput, Tuple]:
        hidden_states = self.embedding_hidden_mapping_in(hidden_states)
        all_hidden_states = (hidden_states, ) if output_hidden_states else None
        all_attentions = () if output_attentions else None

        head_mask = [None] * self.config.num_hidden_layers \
            if head_mask is None else head_mask  # type: ignore

        for i in range(self.config.num_hidden_layers):
            layers_per_group = int(
                self.config.num_hidden_layers / self.config.num_hidden_groups
            )
            group_idx = int(i / (
                self.config.num_hidden_layers / self.config.num_hidden_groups
            ))
            layer_group_output = self.albert_layer_groups[group_idx](
                hidden_states,
                attention_mask,
                head_mask[  # type: ignore
                    group_idx * layers_per_group:
                    (group_idx + 1) * layers_per_group
                ],
                output_attentions,
                output_hidden_states,
            )
            hidden_states = layer_group_output[0]

            if output_attentions:
                all_attentions = all_attentions + layer_group_output[-1]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states, )  # type: ignore  # noqa: E501

        if not return_dict:
            return tuple(
                v for v in [hidden_states, all_hidden_states, all_attentions]
                if v is not None
            )
        else:
            return BaseModelOutput(
                last_hidden_state=hidden_states,  # type: ignore
                hidden_states=all_hidden_states,  # type: ignore
                attentions=all_attentions,
            )
