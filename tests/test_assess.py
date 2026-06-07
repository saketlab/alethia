"""Tests for label-free model assessment."""

import numpy as np

from alethia.assess import (
    assess_models,
    assessment_table,
    generate_positive_pairs,
    make_dirty_variant,
)
from alethia.assess import metrics as metrics_mod

from ._helpers import char_bag, random_embed

REFS = [
    "New York",
    "Los Angeles",
    "Chicago",
    "Houston",
    "Phoenix",
    "Philadelphia",
    "San Antonio",
    "San Diego",
    "Dallas",
    "San Jose",
]
DIRTY = ["New Yrok", "Los Anglees", "Chicagoo", "Houston", "Pheonix"]


class TestSimulator:
    def test_deterministic(self):
        s1, v1 = generate_positive_pairs(REFS, seed=42)
        s2, v2 = generate_positive_pairs(REFS, seed=42)
        assert (s1, v1) == (s2, v2)

    def test_variants_differ_from_source(self):
        src, var = generate_positive_pairs(REFS, n_variants=3, seed=1)
        assert len(src) == len(var) and len(src) > 0
        assert all(s != v for s, v in zip(src, var))

    def test_make_dirty_variant_nonempty(self):
        import random

        rng = random.Random(0)
        for ref in REFS:
            assert make_dirty_variant(ref, rng, n_edits=2)


class TestMetrics:
    def test_separability_orientation(self):
        # char_bag separates city names better than pure noise on a shared seed space.
        sep_good = metrics_mod.reference_separability(char_bag(REFS))
        assert 0.0 <= sep_good["confusability_rate"] <= 1.0
        assert "mean_nn_similarity" in sep_good

    def test_alignment_lower_for_structured_model(self):
        src, var = generate_positive_pairs(REFS, seed=3)
        a_good = metrics_mod.alignment_loss(char_bag(src), char_bag(var))
        a_bad = metrics_mod.alignment_loss(random_embed(src), random_embed(var))
        # A model robust to typos keeps positives closer.
        assert a_good < a_bad

    def test_uniformity_returns_float(self):
        u = metrics_mod.uniformity_loss(char_bag(REFS))
        assert np.isfinite(u)

    def test_margin_and_hubness_keys(self):
        q, r = char_bag(DIRTY), char_bag(REFS)
        assert "mean_margin" in metrics_mod.retrieval_margin(q, r)
        assert "hubness_skew" in metrics_mod.hubness(q, r)
        assert 0.0 <= metrics_mod.mutual_nn_rate(q, r) <= 1.0


class TestAssessor:
    def test_ranks_structured_above_noise(self):
        report = assess_models(
            DIRTY, REFS, {"char_bag": char_bag, "random": random_embed}
        )
        assert report.best is not None
        assert report.best.name == "char_bag"

    def test_table_is_sorted_by_score(self):
        df = assessment_table(
            DIRTY, REFS, {"char_bag": char_bag, "random": random_embed}
        )
        assert list(df["model"])[0] == "char_bag"
        assert df["score"].is_monotonic_decreasing

    def test_per_model_error_is_isolated(self):
        def broken(texts):
            raise RuntimeError("boom")

        report = assess_models(DIRTY, REFS, {"char_bag": char_bag, "broken": broken})
        errs = {a.name: a.error for a in report.assessments}
        assert errs["broken"] is not None
        assert errs["char_bag"] is None
        assert report.best.name == "char_bag"

    def test_html_report_renders(self):
        report = assess_models(
            DIRTY, REFS, {"char_bag": char_bag, "random": random_embed}
        )
        html = report.to_html()
        assert "<html" in html.lower()
        assert "char_bag" in html
        assert "Recommended model" in html

    def test_single_model_is_not_comparative(self):
        report = assess_models(DIRTY, REFS, {"char_bag": char_bag})
        assert not report.is_comparative
        # Composite score is undefined for a single model, but it is still returned.
        assert np.isnan(report.assessments[0].score)
        assert report.best is not None
        assert report.best.name == "char_bag"

    def test_single_model_html_renders_without_recommendation(self):
        report = assess_models(DIRTY, REFS, {"char_bag": char_bag})
        html = report.to_html()
        assert "char_bag" in html
        assert "Recommended model" not in html
        assert "assess two or more" in html

    def test_two_models_are_comparative(self):
        report = assess_models(
            DIRTY, REFS, {"char_bag": char_bag, "random": random_embed}
        )
        assert report.is_comparative
