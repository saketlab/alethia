"""Tests for the assessor meta-evaluation (validation harness)."""

import pytest

from alethia.assess import LabeledDataset, true_accuracy, validate_assessor
from alethia.assess.validate import _kendall_tau, _spearman_rho
from alethia.embedder import as_embedder

from ._helpers import REFS, char_bag, random_embed


def _dataset(name, seed):
    import random

    from alethia.assess.simulate import make_dirty_variant

    rng = random.Random(seed)
    truth = list(REFS)
    queries = [make_dirty_variant(r, rng, 2) for r in truth]
    return LabeledDataset(name, queries, REFS, truth)


class TestLabeledDataset:
    def test_validates_truth_in_references(self):
        with pytest.raises(ValueError):
            LabeledDataset("bad", ["a"], REFS, ["Not A Reference"])

    def test_validates_length(self):
        with pytest.raises(ValueError):
            LabeledDataset("bad", ["a", "b"], REFS, ["New York"])


class TestTrueAccuracy:
    def test_perfect_on_exact(self):
        ds_truth = REFS
        emb = as_embedder(char_bag)
        acc = true_accuracy(REFS, REFS, ds_truth, emb)
        assert acc["top1"] == 1.0
        assert acc["mrr"] == 1.0

    def test_char_bag_beats_random_on_typos(self):
        ds = _dataset("typos", seed=1)
        good = true_accuracy(ds.queries, ds.references, ds.truth, as_embedder(char_bag))
        bad = true_accuracy(
            ds.queries, ds.references, ds.truth, as_embedder(random_embed)
        )
        assert good["top1"] >= bad["top1"]


class TestRankCorrelations:
    def test_kendall_perfect_agreement(self):
        assert _kendall_tau([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)

    def test_kendall_perfect_disagreement(self):
        assert _kendall_tau([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)

    def test_spearman_perfect_agreement(self):
        assert _spearman_rho([1, 2, 3, 4], [5, 6, 7, 8]) == pytest.approx(1.0)


class TestValidateAssessor:
    def test_label_free_ranks_match_truth(self):
        datasets = [_dataset("d1", seed=1), _dataset("d2", seed=2)]
        models = {"char_bag": char_bag, "random": random_embed}
        result = validate_assessor(datasets, models, metric="top1")
        assert result.kendall_tau > 0
        assert result.n_datasets == 2
        assert result.n_models == 2

    def test_table_has_true_and_predicted_columns(self):
        result = validate_assessor([_dataset("d1", seed=1)], {"char_bag": char_bag})
        df = result.to_table()
        for col in ("dataset", "model", "true_top1", "true_mrr", "label_free_score"):
            assert col in df.columns

    def test_invalid_metric_rejected(self):
        with pytest.raises(ValueError):
            validate_assessor(
                [_dataset("d1", seed=1)], {"char_bag": char_bag}, metric="f1"
            )


class TestOpenWorldNIL:
    """A matcher that cannot decline is not deployable; the harness must be able to ask."""

    def _nil_dataset(self):
        refs = list(REFS)
        return LabeledDataset(
            "nil",
            queries=["New Yrok", "Chicagoo", "Ulaanbaatar", "Tegucigalpa"],
            references=refs,
            truth=["New York", "Chicago", None, None],
        )

    def test_none_truth_is_accepted_and_flagged(self):
        ds = self._nil_dataset()
        assert ds.has_nil
        assert ds.nil_rate == pytest.approx(0.5)

    def test_dataset_without_nil_is_unchanged(self):
        ds = _dataset("plain", seed=1)
        assert not ds.has_nil
        assert ds.nil_rate == 0.0

    def test_truth_still_validated_against_references(self):
        with pytest.raises(ValueError):
            LabeledDataset("bad", ["a", "b"], REFS, ["New York", "Not A Reference"])

    def test_threshold_required_when_nil_present(self):
        ds = self._nil_dataset()
        with pytest.raises(ValueError, match="nil_threshold"):
            true_accuracy(ds.queries, ds.references, ds.truth, as_embedder(char_bag))

    def test_abstention_is_scored(self):
        ds = self._nil_dataset()
        emb = as_embedder(char_bag)
        always = true_accuracy(
            ds.queries, ds.references, ds.truth, emb, nil_threshold=1.1
        )
        assert always["nil_recall"] == pytest.approx(1.0)
        assert always["answerable_top1"] == pytest.approx(0.0)

        never = true_accuracy(
            ds.queries, ds.references, ds.truth, emb, nil_threshold=-1.1
        )
        assert never["nil_recall"] == pytest.approx(0.0)
        assert never["answerable_top1"] > 0.0

    def test_mrr_ignores_nil_queries(self):
        ds = self._nil_dataset()
        acc = true_accuracy(
            ds.queries,
            ds.references,
            ds.truth,
            as_embedder(char_bag),
            nil_threshold=0.5,
        )
        assert 0.0 < acc["mrr"] <= 1.0


class TestSpearmanTies:
    def test_ties_get_mid_ranks(self):
        """Ordinal ranks made rho depend on the sort's tie order and disagree with scipy."""
        from alethia.assess.validate import _spearman_rho

        assert _spearman_rho([1, 1, 1, 2], [3, 1, 2, 4]) == pytest.approx(0.7745966692)
        assert _spearman_rho([1, 1, 3, 4], [2, 2, 3, 4]) == pytest.approx(1.0)

    def test_constant_input_is_undefined_not_zero(self):
        import numpy as np

        from alethia.assess.validate import _spearman_rho

        assert np.isnan(_spearman_rho([0.5] * 4, [1, 2, 3, 4]))
