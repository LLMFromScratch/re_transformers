import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.pytorch_utils import apply_chunking_to_forward

from .configuration_albert import AlbertConfig
from ...modeling_outputs import BaseModelOutput


class AlbertEmbeddings(nn.Module):
    def __init__(self, config: AlbertConfig) -> None:
        super().__init__()


class AlbertAttention(nn.Module):
    def __init__(self, config: AlbertConfig) -> None:
        super().__init__()

        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads}"
            )

        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention_dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.output_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * self.max_position_embeddings - 1, self.attention_head_size)

        self.pruned_heads = set()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_length, _ = hidden_states.shape

        query_layer: torch.Tensor = self.query(hidden_states)
        key_layer: torch.Tensor = self.key(hidden_states)
        value_layer: torch.Tensor = self.value(hidden_states)

        query_layer = query_layer.view(batch_size, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        key_layer = key_layer.view(batch_size, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        value_layer = value_layer.view(batch_size, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding: torch.Tensor = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            if self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose(1, 2).flatten(2)

        projected_context_layer = self.dense(context_layer)
        projected_context_layer_dropout = self.output_dropout(projected_context_layer)
        layernormed_context_layer = self.LayerNorm(projected_context_layer_dropout + hidden_states)

        return (layernormed_context_layer, attention_probs) if output_attentions else (layernormed_context_layer, )


class AlbertSdpaAttention(AlbertAttention):
    def __init__(self, config: AlbertConfig) -> None:
        super().__init__(config)

        self.dropout_prob = config.attention_probs_dropout_prob

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        if self.position_embedding_type != "absolute" or output_attentions:
            return super().forward(hidden_states, attention_mask, head_mask, output_attentions)

        batch_size, seq_len, _ = hidden_states.shape

        query_layer: torch.Tensor = self.query(hidden_states)
        key_layer: torch.Tensor = self.key(hidden_states)
        value_layer: torch.Tensor = self.value(hidden_states)

        query_layer = query_layer.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        key_layer = key_layer.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        value_layer = value_layer.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)

        attention_output = F.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
            self.dropout_prob if self.training else 0.0,
            is_causal=False,
        )

        attention_output = attention_output.transpose(1, 2).flatten(2)

        projected_context_layer = self.dense(attention_output)
        projected_context_layer_dropout = self.output_dropout(projected_context_layer)
        layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)

        return (layernormed_context_layer, )


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

        self.attention = ALBERT_ATTENTION_CLASSES[config._attn_implementation](config)  # type: ignore
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
                all_hidden_states = all_hidden_states + (hidden_states, )  # type: ignore

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


# Questions
# 1. `head_mask`
# 2. `position_embedding_type`
# 3. Why are `position_embeddings` and `token_type_embeddings` not padded?
