import unittest

import torch
from transformers import AlbertForMaskedLM, AlbertModel
from transformers.models.albert.modeling_albert import AlbertMLMHead

from re_transformers import (
    AlbertForMaskedLM as re_AlbertForMaskedLM,
    AlbertMLMHead as re_AlbertMLMHead,
    AlbertModel as re_AlbertModel
)


class TestModelingAlbert(unittest.TestCase):
    def setUp(self) -> None:
        # Device config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Base model
        self.name = "albert/albert-base-v2"
        self.model: AlbertModel = AlbertModel.from_pretrained(
            self.name).to(device=self.device).eval()
        self.re_model: re_AlbertModel = re_AlbertModel.from_pretrained(
            self.name).to(device=self.device).eval()

        # Model config
        self.config = self.model.config
        self.dtype = torch.float32

        # Model head
        self.masked_lm_model: AlbertForMaskedLM = AlbertForMaskedLM.from_pretrained(
            self.name).to(device=self.device).eval()
        self.mlm_head: AlbertMLMHead = self.masked_lm_model.predictions

        self.re_masked_lm_model: re_AlbertForMaskedLM = re_AlbertForMaskedLM.from_pretrained(
            self.name).to(device=self.device).eval()
        self.re_mlm_head: re_AlbertMLMHead = self.re_masked_lm_model.predictions

        # Input config
        self.batch_size = 1
        self.seq_length = 10
        self.input_shape = (self.batch_size, self.seq_length)
        self.hidden_states_shape = (
            self.batch_size, self.seq_length, self.config.hidden_size)

        # Default inputs
        self.input_ids = torch.randint(
            low=0, high=self.config.vocab_size, size=self.input_shape, dtype=torch.long, device=self.device)
        self.token_type_ids = torch.randint(
            low=0, high=self.config.type_vocab_size, size=self.input_shape, dtype=torch.long, device=self.device)
        self.position_ids = torch.arange(
            self.seq_length, dtype=torch.long, device=self.device).unsqueeze(0)
        self.hidden_states = torch.rand(
            self.hidden_states_shape, dtype=self.dtype, device=self.device)

    def test_AlbertEmbeddings(self) -> None:
        embeddings = self.model.embeddings
        re_embeddings = self.re_model.embeddings

        expected = embeddings(
            input_ids=self.input_ids, token_type_ids=self.token_type_ids, position_ids=self.position_ids)
        actual = re_embeddings(
            input_ids=self.input_ids, token_type_ids=self.token_type_ids, position_ids=self.position_ids)
        assert torch.equal(actual, expected)

    def test_AlbertAttention(self) -> None:
        attention = self.model.encoder.albert_layer_groups[0].albert_layers[0].attention
        re_attention = self.re_model.encoder.albert_layer_groups[0].albert_layers[0].attention

        expected = attention(hidden_states=self.hidden_states)[0]
        actual = re_attention(hidden_states=self.hidden_states)[0]
        assert torch.equal(actual, expected)

    def test_AlbertLayer(self) -> None:
        layer = self.model.encoder.albert_layer_groups[0].albert_layers[0]
        re_layer = self.re_model.encoder.albert_layer_groups[0].albert_layers[0]

        expected = layer(hidden_states=self.hidden_states)[0]
        actual = re_layer(hidden_states=self.hidden_states)[0]
        assert torch.equal(actual, expected)

    def test_AlbertTransformer(self) -> None:
        transformer_hidden_states = torch.rand(
            (self.batch_size, self.seq_length, self.config.embedding_size), dtype=self.dtype, device=self.device)

        transformer = self.model.encoder
        re_transformer = self.re_model.encoder

        expected = transformer(hidden_states=transformer_hidden_states)[0]
        actual = re_transformer(hidden_states=transformer_hidden_states)[0]
        assert torch.equal(actual, expected)

    def test_AlbertModel(self) -> None:
        expected_sequence_output, expected_pooled_output = self.model(
            input_ids=self.input_ids, token_type_ids=self.token_type_ids, position_ids=self.position_ids)[:2]
        actual_sequence_output, actual_pooled_output = self.re_model(input_ids=self.input_ids,
                                                                     token_type_ids=self.token_type_ids, position_ids=self.position_ids)[:2]
        assert torch.equal(actual_sequence_output, expected_sequence_output)
        assert torch.equal(actual_pooled_output, expected_pooled_output)

    def test_AlbertMLMHead(self) -> None:
        expected = self.mlm_head(hidden_states=self.hidden_states)
        actual = self.re_mlm_head(hidden_states=self.hidden_states)
        assert torch.equal(actual, expected)

    def test_AlbertForMaskedLM(self) -> None:
        expected = self.masked_lm_model(
            input_ids=self.input_ids, token_type_ids=self.token_type_ids, position_ids=self.position_ids)[0]
        actual = self.re_masked_lm_model(
            input_ids=self.input_ids, token_type_ids=self.token_type_ids, position_ids=self.position_ids)[0]
        assert torch.equal(actual, expected)
