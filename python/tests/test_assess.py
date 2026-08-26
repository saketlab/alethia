"""Tests for label-free model assessment."""

import numpy as np

from alethia.assess import (
    assess_models,
    assessment_table,
    generate_positive_pairs,
    make_dirty_variant,
)
from alethia.assess import metrics as metrics_mod

from ._helpers import REFS, char_bag, random_embed

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
        sep_good = metrics_mod.reference_separability(char_bag(REFS))
        assert 0.0 <= sep_good["confusability_rate"] <= 1.0
        assert "mean_nn_similarity" in sep_good

    def test_alignment_lower_for_structured_model(self):
        src, var = generate_positive_pairs(REFS, seed=3)
        a_good = metrics_mod.alignment_loss(char_bag(src), char_bag(var))
        a_bad = metrics_mod.alignment_loss(random_embed(src), random_embed(var))
        assert a_good < a_bad  # a model robust to typos keeps positives closer

    def test_positive_pair_rank_prefers_structured_model(self):
        src, var = generate_positive_pairs(REFS, seed=3)
        a_good = metrics_mod.positive_pair_rank(char_bag(src), char_bag(var))
        a_bad = metrics_mod.positive_pair_rank(random_embed(src), random_embed(var))
        assert a_good > a_bad

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


import pytest  # noqa: E402

from alethia.assess import metrics, simulate  # noqa: E402


class TestCompositeMissingMetric:
    """A model that fails to produce a metric must not outrank one that scored badly."""

    def test_missing_metric_is_imputed_at_worst_value(self):
        from alethia.assess.assessor import ModelAssessment, _composite_scores

        good = ModelAssessment("good", {"alignment_loss": 0.10})
        bad = ModelAssessment("bad", {"alignment_loss": 0.90})
        absent = ModelAssessment("absent", {})
        _composite_scores(
            [good, bad, absent], {"alignment_loss": 1.5}, family_normalize=False
        )

        assert good.score > bad.score
        assert absent.score <= bad.score


class TestFamilyNormalizedWeights:
    """Adding correlated metrics to a family must not inflate that family's influence."""

    def test_each_family_sums_to_its_family_weight(self):
        from collections import defaultdict

        from alethia.assess.assessor import (
            _DEFAULT_WEIGHTS,
            _FAMILY_WEIGHTS,
            _METRICS,
            _family_normalized_weights,
        )

        scaled = _family_normalized_weights(_DEFAULT_WEIGHTS, list(_DEFAULT_WEIGHTS))
        totals = defaultdict(float)
        for key, w in scaled.items():
            totals[_METRICS[key].family] += w

        for family, total in totals.items():
            assert total == pytest.approx(_FAMILY_WEIGHTS[family])

    def test_absent_family_drops_out_without_disturbing_others(self):
        from alethia.assess.assessor import (
            _DEFAULT_WEIGHTS,
            _METRICS,
            _family_normalized_weights,
        )

        ref_only = [
            k
            for k, spec in _METRICS.items()
            if spec.family in ("separability", "geometry", "robustness")
        ]
        scaled = _family_normalized_weights(_DEFAULT_WEIGHTS, ref_only)

        assert all(_METRICS[k].family != "retrieval" for k in scaled)
        separability = sum(
            w for k, w in scaled.items() if _METRICS[k].family == "separability"
        )
        assert separability == pytest.approx(2.0)


class TestBlockedMetrics:
    """Blocked implementations must match the naive implementations."""

    def test_nearest_other_similarity_matches_full_matrix(self):
        rng = np.random.default_rng(0)
        emb = metrics.l2_normalize(rng.normal(size=(300, 16)))

        sims = emb @ emb.T
        np.fill_diagonal(sims, -np.inf)
        expected = sims.max(axis=1)

        for block_rows in (7, 64, 4096):
            got = metrics._nearest_other_similarity(emb, block_rows=block_rows)
            assert np.allclose(expected, got)

    def test_mutual_nn_rate_matches_dense_formulation(self):
        rng = np.random.default_rng(1)
        q = rng.normal(size=(120, 12))
        r = rng.normal(size=(200, 12))

        qn, rn = metrics.l2_normalize(q), metrics.l2_normalize(r)
        qr = qn @ rn.T
        top1 = np.argmax(qr, axis=1)
        chosen = qr.T[top1]
        topk = np.argpartition(-chosen, kth=4, axis=1)[:, :5]
        expected = float(np.any(topk == np.arange(120)[:, None], axis=1).sum() / 120)

        assert metrics.mutual_nn_rate(q, r) == pytest.approx(expected)

    def test_hubness_matches_dense_formulation_and_shared_sims(self):
        rng = np.random.default_rng(19)
        q = rng.normal(size=(73, 9))
        r = rng.normal(size=(101, 9))
        qn, rn = metrics.l2_normalize(q), metrics.l2_normalize(r)
        sims = qn @ rn.T
        idx = np.argpartition(-sims, kth=4, axis=1)[:, :5]
        counts = np.bincount(idx.ravel(), minlength=len(r)).astype(float)
        expected = np.mean(((counts - counts.mean()) / counts.std()) ** 3)

        assert metrics.hubness(q, r)["hubness_skew"] == pytest.approx(expected)
        assert metrics.hubness(qn, rn, _normalized=True, _sims=sims)[
            "hubness_skew"
        ] == pytest.approx(expected)

    def test_pre_normalized_fast_paths_match_public_paths(self):
        rng = np.random.default_rng(23)
        q = rng.normal(size=(31, 12))
        r = rng.normal(size=(67, 12))
        qn, rn = metrics.l2_normalize(q), metrics.l2_normalize(r)
        sims = qn @ rn.T

        assert metrics.reference_separability(r) == pytest.approx(
            metrics.reference_separability(rn, _normalized=True)
        )
        assert metrics.uniformity_loss(r) == pytest.approx(
            metrics.uniformity_loss(rn, _normalized=True)
        )
        assert metrics.retrieval_margin(q, r) == pytest.approx(
            metrics.retrieval_margin(qn, rn, _normalized=True, _sims=sims)
        )
        assert metrics.mutual_nn_rate(q, r) == pytest.approx(
            metrics.mutual_nn_rate(qn, rn, _normalized=True, _sims=sims)
        )

    def test_intrinsic_dimensionality_matches_svd_definition(self):
        rng = np.random.default_rng(29)
        emb = rng.normal(size=(500, 31)) * rng.uniform(0.2, 2.0, (500, 1))
        centered = emb - emb.mean(axis=0, keepdims=True)
        lam = np.linalg.svd(centered, compute_uv=False) ** 2
        expected = (lam.sum() ** 2) / np.sum(lam**2)

        got = metrics.intrinsic_dimensionality(emb)
        assert got["participation_ratio"] == pytest.approx(expected, rel=1e-12)
        assert got["normalized_pr"] == pytest.approx(expected / emb.shape[1], rel=1e-12)


class TestAnisotropyCorrection:
    """Centering must strip the cone effect without changing real discriminability."""

    def test_margin_z_is_invariant_to_an_added_common_direction(self):
        rng = np.random.default_rng(2)
        base = rng.normal(size=(300, 24))
        isotropic = metrics.l2_normalize(base)
        anisotropic = metrics.l2_normalize(base + 6.0)  # push into a narrow cone

        raw_iso = metrics.reference_separability(isotropic)["mean_nn_similarity"]
        raw_ani = metrics.reference_separability(anisotropic)["mean_nn_similarity"]
        assert raw_ani > raw_iso + 0.3  # raw cosine is badly fooled

        z_iso = metrics.centered_separability(isotropic)["nn_margin_z"]
        z_ani = metrics.centered_separability(anisotropic)["nn_margin_z"]
        assert z_ani == pytest.approx(z_iso, abs=0.15)  # corrected view is not

    def test_order_preserving_cone_transform_only_changes_absolute_metrics(self):
        rng = np.random.default_rng(17)
        refs = metrics.l2_normalize(rng.normal(size=(120, 20)))
        queries = metrics.l2_normalize(rng.normal(size=(40, 20)))
        variants = metrics.l2_normalize(refs + 0.1 * rng.normal(size=refs.shape))

        def cone(x):
            return metrics.l2_normalize(np.column_stack((x, np.full(x.shape[0], 6.0))))

        cone_refs, cone_queries, cone_variants = (
            cone(refs),
            cone(queries),
            cone(variants),
        )
        # for normalized inputs this is exactly (cosine + c**2) / (1 + c**2)
        assert np.array_equal(
            np.argsort(queries @ refs.T, axis=1),
            np.argsort(cone_queries @ cone_refs.T, axis=1),
        )
        assert np.array_equal(
            np.argsort(refs @ variants.T, axis=1),
            np.argsort(cone_refs @ cone_variants.T, axis=1),
        )

        raw = metrics.reference_separability(refs)
        raw_cone = metrics.reference_separability(cone_refs)
        assert raw_cone["mean_nn_similarity"] > raw["mean_nn_similarity"] + 0.4
        assert metrics.uniformity_loss(cone_refs) > metrics.uniformity_loss(refs)
        assert metrics.alignment_loss(
            cone_refs, cone_variants
        ) < metrics.alignment_loss(refs, variants)

        centered = metrics.centered_separability(refs)
        centered_cone = metrics.centered_separability(cone_refs)
        assert centered_cone["centered_nn_similarity"] == pytest.approx(
            centered["centered_nn_similarity"], abs=1e-12
        )
        assert centered_cone["nn_margin_z"] == pytest.approx(
            centered["nn_margin_z"], abs=1e-12
        )
        assert metrics.positive_pair_rank(cone_refs, cone_variants) == pytest.approx(
            metrics.positive_pair_rank(refs, variants), abs=1e-12
        )
        margin = metrics.retrieval_margin(queries, refs)
        margin_cone = metrics.retrieval_margin(cone_queries, cone_refs)
        assert margin_cone["mean_margin_z"] == pytest.approx(
            margin["mean_margin_z"], abs=1e-12
        )
        assert margin_cone["low_margin_z_rate"] == pytest.approx(
            margin["low_margin_z_rate"], abs=1e-12
        )


class TestNoiseProfileEstimation:
    """The perturbation mix should be measured from the data, not assumed."""

    def test_uppercase_queries_produce_an_uppercase_weighted_profile(self):
        queries = [
            "ACUTE GASTROENTERITIS",
            "SEVERE SEPSIS",
            "CHRONIC LIVER DISEASE",
        ] * 10
        references = [
            "Acute gastroenteritis",
            "Severe sepsis",
            "Chronic liver disease",
        ] * 10

        profile = simulate.estimate_noise_profile(queries, references)

        assert profile.weights["uppercase"] > profile.weights["typo"]
        assert profile.evidence["query_upper_rate"] > 0.9
        assert profile.evidence["reference_upper_rate"] == pytest.approx(0.0)

    def test_comma_clauses_are_dropped_when_queries_never_carry_them(self):
        references = ["Cholera, unspecified", "Typhoid fever, acute"] * 20
        queries = ["cholera", "typhoid fever"] * 20

        profile = simulate.estimate_noise_profile(queries, references)
        assert profile.weights.get("drop_comma_clause", 0) > 0

        _, variants = simulate.generate_positive_pairs(
            references, n_variants=4, seed=0, profile=profile
        )
        assert any("," not in v for v in variants)

    def test_empty_queries_fall_back_to_the_default_profile(self):
        profile = simulate.estimate_noise_profile([], ["Cholera"])
        assert profile.weights == simulate._DEFAULT_PROFILE_WEIGHTS

    def test_make_dirty_variant_is_deterministic_under_a_profile(self):
        import random

        profile = simulate.NoiseProfile({"uppercase": 1.0, "typo": 1.0})
        a = simulate.make_dirty_variant(
            "Cholera due to Vibrio", random.Random(7), 3, profile
        )
        b = simulate.make_dirty_variant(
            "Cholera due to Vibrio", random.Random(7), 3, profile
        )
        assert a == b


class TestScaleFreeRetrievalMargin:
    """The raw cosine margin is anisotropy-confounded; the z variant must not be."""

    def test_z_margin_survives_a_cone_transform_that_crushes_the_raw_margin(self):
        rng = np.random.default_rng(0)
        refs = rng.normal(size=(400, 32))
        queries = rng.normal(size=(60, 32))

        iso = metrics.retrieval_margin(
            metrics.l2_normalize(queries), metrics.l2_normalize(refs)
        )
        ani = metrics.retrieval_margin(
            metrics.l2_normalize(queries + 6.0), metrics.l2_normalize(refs + 6.0)
        )

        raw_ratio = iso["mean_margin"] / ani["mean_margin"]
        z_ratio = iso["mean_margin_z"] / ani["mean_margin_z"]
        assert raw_ratio > 5.0  # the raw margin collapses
        assert z_ratio < 2.0  # the z margin roughly holds

    def test_z_variants_are_reported_and_weighted_above_the_raw_ones(self):
        from alethia.assess.assessor import _DEFAULT_WEIGHTS, _METRICS

        assert _METRICS["mean_margin_z"].family == "retrieval"
        assert _DEFAULT_WEIGHTS["mean_margin_z"] > _DEFAULT_WEIGHTS["mean_margin"]
        assert (
            _DEFAULT_WEIGHTS["low_margin_z_rate"] > _DEFAULT_WEIGHTS["low_margin_rate"]
        )

    def test_default_composite_excludes_absolute_cosine_and_distance_metrics(self):
        from alethia.assess.assessor import _DEFAULT_WEIGHTS

        for key in (
            "mean_nn_similarity",
            "confusability_rate",
            "alignment_loss",
            "uniformity_loss",
            "mean_margin",
            "low_margin_rate",
        ):
            assert _DEFAULT_WEIGHTS[key] == 0.0
