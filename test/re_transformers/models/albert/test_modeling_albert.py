import unittest

import torch
from transformers import AlbertModel

from re_transformers import AlbertModel as re_AlbertModel


class TestModelingAlbert(unittest.TestCase):
    def setUp(self):
        # Device config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Input config
        self.batch_size = 1
        self.seq_length = 10
        self.input_shape = (self.batch_size, self.seq_length)
        self.hidden_states_shape = (
            self.batch_size, self.seq_length, self.config.hidden_size)

        # Base model
        self.model: AlbertModel = AlbertModel.from_pretrained(
            "albert/albert-base-v2").to(device=self.device)
        self.re_model: re_AlbertModel = re_AlbertModel.from_pretrained(
            "albert/albert-base-v2").to(device=self.device)

        self.config = self.model.config

    def test_AlbertEmbeddings(self):
        input_ids = torch.randint(
            low=0,
            high=self.config.vocab_size,
            size=self.input_shape,
            dtype=torch.long,
            device=self.device,
        )
        token_type_ids = torch.randint(
            low=0,
            high=self.config.type_vocab_size,
            size=self.input_shape,
            dtype=torch.long,
            device=self.device,
        )
        position_ids = torch.arange(
            self.seq_length, dtype=torch.long, device=self.device).unsqueeze(0)

        embeddings = self.model.embeddings
        re_embeddings = self.re_model.embeddings

        expected = embeddings(input_ids, token_type_ids, position_ids)
        actual = re_embeddings(input_ids, position_ids, token_type_ids)
        assert torch.equal(actual, expected)

    def test_AlbertAttention(self):
        hidden_states = torch.rand(
            self.hidden_states_shape, dtype=torch.float32, device=self.device)

        attention = self.model.encoder.albert_layer_groups[0].albert_layers[0].attention
        re_attention = self.re_model.encoder.albert_layer_groups[0].albert_layers[0].attention
