from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from .configuration_albert import AlbertConfig
from ...modeling_outputs import BaseModelOutput


class AlbertLayer(nn.Module):
    def __init__(self, config: AlbertConfig):
        super().__init__()


class AlbertLayerGroup(nn.Module):
    def __init__(self, config: AlbertConfig):
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
    def __init__(self, config: AlbertConfig):
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
