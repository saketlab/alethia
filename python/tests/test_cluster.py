"""Tests for embedding-based entity clustering (mutual-NN + margin)."""

import numpy as np
import pytest

from alethia import cluster_entities, mutual_nn_edges
from alethia.cluster import ClusterResult

from ._helpers import char_bag, random_embed


class TestMutualNNEdges:
    def test_identical_strings_dedupe_before_edges(self):
        res = cluster_entities(["abc", "abc", "abc"], char_bag, floor=0.5)
        assert res.entities == ["abc"]
        assert res.n_clusters() == 1

    def test_near_duplicates_merge(self):
        ents = ["color", "colour", "zzzzzzzzz"]
        res = cluster_entities(ents, char_bag, floor=0.5, k=2)
        groups = res.clusters()
        sizes = sorted(len(m) for m in groups.values())
        assert sizes == [1, 2]

    def test_edges_are_symmetric_pairs(self):
        emb = char_bag(["alpha", "alpha2", "totally different string here"])
        edges = mutual_nn_edges(emb, floor=0.3, k=2)
        for e in edges:
            assert e.i < e.j
            assert isinstance(e.mutual, (bool, np.bool_))

    def test_require_mutual_filters_one_directional(self):
        emb = char_bag(["aaa", "aaab", "aaabb", "qqqqq"])
        strict = mutual_nn_edges(emb, floor=0.3, k=3, require_mutual=True)
        loose = mutual_nn_edges(emb, floor=0.3, k=3, require_mutual=False)
        assert len(loose) >= len(strict)
        assert all(e.mutual for e in strict)

    def test_floor_excludes_dissimilar(self):
        emb = char_bag(["abc", "xyz"])
        edges = mutual_nn_edges(emb, floor=0.99, k=1)
        assert edges == []


class TestConfidence:
    def test_confidence_folds_margin(self):
        emb = char_bag(["report alpha", "report alpha v2", "report beta"])
        edges = mutual_nn_edges(emb, floor=0.3, k=2)
        for e in edges:
            expected = e.cosine * (1.0 + min(e.margin_i, e.margin_j))
            assert abs(e.confidence - expected) < 1e-6

    def test_edge_records_sorted_by_confidence(self):
        ents = ["new york", "new york city", "los angeles", "los angeles ca"]
        res = cluster_entities(ents, char_bag, floor=0.3, k=3)
        recs = res.edge_records()
        confs = [r["confidence"] for r in recs]
        assert confs == sorted(confs, reverse=True)


class TestCanonical:
    def test_shortest_canonical(self):
        ents = ["abc", "abcd", "abcde"]
        res = cluster_entities(ents, char_bag, floor=0.5, k=2, canonical="shortest")
        for cid, members in res.clusters().items():
            if len(members) > 1:
                assert res.canonical[cid] == min(members, key=len)

    def test_first_canonical(self):
        ents = ["abcd", "abc"]
        res = cluster_entities(ents, char_bag, floor=0.5, k=1, canonical="first")
        for cid, members in res.clusters().items():
            if len(members) > 1:
                assert res.canonical[cid] == members[0]


class TestResultShape:
    def test_to_records_roundtrip(self):
        ents = ["foo", "foo bar", "baz"]
        res = cluster_entities(ents, char_bag, floor=0.4, k=2)
        recs = res.to_records()
        assert len(recs) == len(set(ents))
        assert {"entity", "cluster", "canonical"} == set(recs[0])

    def test_bad_embedder_produces_more_singletons(self):
        ents = ["color", "colour", "flavour", "flavor"]
        good = cluster_entities(ents, char_bag, floor=0.6, k=3)
        bad = cluster_entities(ents, random_embed, floor=0.6, k=3)
        assert bad.n_clusters() >= good.n_clusters()

    def test_single_entity(self):
        res = cluster_entities(["solo"], char_bag)
        assert isinstance(res, ClusterResult)
        assert res.n_clusters() == 1
        assert res.edges == []

    def test_min_confidence_drops_weak_edges(self):
        ents = ["color", "colour", "zzzzzzzzz"]
        keep = cluster_entities(ents, char_bag, floor=0.5)
        drop = cluster_entities(ents, char_bag, floor=0.5, min_confidence=99.0)
        assert keep.n_clusters() < drop.n_clusters()

    def test_non_finite_embeddings_raise(self):
        def nan_embed(texts):
            arr = char_bag(texts)
            arr[0, 0] = np.nan
            return arr

        with pytest.raises(ValueError, match="non-finite"):
            cluster_entities(["alpha", "beta"], nan_embed)


class TestEdgeCases:
    def test_two_entities_have_a_finite_confidence(self):
        """With n == 2 the masked row has no runner-up, and inf leaked into the CSV."""
        result = cluster_entities(["New York", "New Yrok"], model=char_bag)
        for row in result.edge_records():
            assert np.isfinite(row["confidence"])
            assert np.isfinite(row["margin"])

    def test_min_confidence_still_filters_two_entities(self):
        far = cluster_entities(
            ["New York", "New Yrok"], model=char_bag, min_confidence=99.0
        )
        assert far.edge_records() == []
