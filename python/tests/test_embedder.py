"""Tests for the model-agnostic embedding interface."""

import numpy as np
import pytest

from alethia import alethia
from alethia.embedder import (
    CallableEmbedder,
    as_embedder,
    match_by_embeddings,
)

from ._helpers import char_bag


class TestAsEmbedder:
    def test_callable_becomes_embedder(self):
        emb = as_embedder(char_bag)
        assert isinstance(emb, CallableEmbedder)
        arr = emb.encode(["abc", "abcd"])
        assert arr.shape[0] == 2
        assert arr.shape[1] == emb.dim

    def test_embedder_passthrough(self):
        emb = CallableEmbedder(char_bag)
        assert as_embedder(emb) is emb

    def test_non_embedding_keyword_rejected(self):
        with pytest.raises(ValueError):
            as_embedder("rapidfuzz")

    def test_unknown_type_rejected(self):
        with pytest.raises(TypeError):
            as_embedder(12345)

    def test_callable_1d_for_many_inputs_rejected(self):
        bad = lambda texts: np.zeros(26, dtype=np.float32)  # noqa: E731
        with pytest.raises(ValueError):
            as_embedder(bad).encode(["a", "b"])


class TestMatchByEmbeddings:
    def test_basic_matching(self):
        emb = as_embedder(char_bag)
        refs = ["New York", "Los Angeles", "Chicago"]
        res = match_by_embeddings(["New Yrok", "Chicagoo"], refs, emb)
        preds = {r["given_entity"]: r["alethia_prediction"] for r in res}
        assert preds["New Yrok"] == "New York"
        assert preds["Chicagoo"] == "Chicago"

    def test_empty_references(self):
        emb = as_embedder(char_bag)
        res = match_by_embeddings(["x"], [], emb)
        assert len(res) == 1
        assert np.isnan(res[0]["alethia_score"])


class TestAlethiaWithCallable:
    def test_alethia_accepts_callable_model(self):
        refs = ["New York", "Los Angeles", "Chicago", "Houston"]
        res = alethia(["New Yrok", "Los Anglees"], refs, model=char_bag)
        assert res.attrs["backend"] == "callable"
        preds = dict(zip(res["given_entity"], res["alethia_prediction"]))
        assert preds["New Yrok"] == "New York"
        assert preds["Los Anglees"] == "Los Angeles"
