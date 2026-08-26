"""Orchestrates per-model label-free metrics into a composite score and report."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..embedder import CachingEmbedder, ModelSpec, as_embedder
from . import metrics as metrics_mod
from .simulate import NoiseProfile, estimate_noise_profile, generate_positive_pairs

logger = logging.getLogger(__name__)


@dataclass
class ModelAssessment:
    """Label-free assessment of a single model on one dataset."""

    name: str
    metrics: dict[str, float] = field(default_factory=dict)
    score: float = float("nan")
    error: str | None = None

    def as_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {"model": self.name, "score": self.score}
        row.update(self.metrics)
        if self.error:
            row["error"] = self.error
        return row


@dataclass
class AssessmentReport:
    """Assessment of several models on one dataset."""

    assessments: list[ModelAssessment]
    n_queries: int
    n_references: int
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def is_comparative(self) -> bool:
        """True when at least two models were assessed, so the composite score is meaningful."""
        return sum(not a.error for a in self.assessments) >= 2

    @property
    def best(self) -> ModelAssessment | None:
        """Highest-scoring model, or the sole assessed model when only one was given."""
        scored = [a for a in self.assessments if not a.error and np.isfinite(a.score)]
        if scored:
            return max(scored, key=lambda a: a.score)
        assessed = [a for a in self.assessments if not a.error]
        return assessed[0] if len(assessed) == 1 else None

    def to_table(self):
        """Return a tidy, score-sorted :class:`pandas.DataFrame` of the assessment."""
        import pandas as pd

        df = pd.DataFrame([a.as_row() for a in self.assessments])
        if "score" in df.columns:
            df = df.sort_values("score", ascending=False, na_position="last")
        return df.reset_index(drop=True)

    def to_html(
        self, path: str | None = None, title: str = "Alethia Model Assessment"
    ) -> str:
        """Render a professional, self-contained HTML report (see :mod:`.report`)."""
        from .report import render_html

        return render_html(self, path=path, title=title)


@dataclass(frozen=True)
class MetricSpec:
    """How one metric enters the composite score.

    Attributes:
        orientation: +1 if higher-is-better, -1 if lower-is-better.
        family: Latent property probed. Correlated views of one property share a family,
            and weights are normalized *within* it.
        weight: Weight within the family. ``0.0`` reports the metric without scoring it.
        transform: Applied before scoring, for metrics whose distance from zero matters.
    """

    orientation: int
    family: str
    weight: float
    transform: Any | None = None


_METRICS: dict[str, MetricSpec] = {
    "mean_nn_similarity": MetricSpec(-1, "separability", 0.0),
    "confusability_rate": MetricSpec(-1, "separability", 0.0),
    "centered_nn_similarity": MetricSpec(-1, "separability", 1.0),
    "nn_margin_z": MetricSpec(+1, "separability", 0.0),
    "normalized_pr": MetricSpec(+1, "geometry", 0.0),
    "uniformity_loss": MetricSpec(-1, "geometry", 0.0),
    "alignment_loss": MetricSpec(-1, "robustness", 0.0),
    "positive_pair_rank": MetricSpec(+1, "robustness", 1.5),
    "mean_margin": MetricSpec(+1, "retrieval", 0.0),
    "low_margin_rate": MetricSpec(-1, "retrieval", 0.0),
    "mean_margin_z": MetricSpec(+1, "retrieval", 1.0),
    "low_margin_z_rate": MetricSpec(-1, "retrieval", 0.5),
    "mutual_nn_rate": MetricSpec(+1, "retrieval", 1.0),
    "hubness_skew": MetricSpec(-1, "pathology", 0.75, transform=np.abs),
}

_FAMILY_WEIGHTS = {
    "separability": 2.0,
    "robustness": 1.5,
    "retrieval": 1.5,
    "geometry": 1.0,
    "pathology": 0.75,
}

_DEFAULT_WEIGHTS = {key: spec.weight for key, spec in _METRICS.items()}


def _family_normalized_weights(
    weights: dict[str, float], available: Sequence[str]
) -> dict[str, float]:
    """Rescale ``weights`` so each metric family contributes exactly its family weight."""
    usable = [k for k in weights if k in available and weights[k] > 0]
    by_family: dict[str, list[str]] = {}
    for key in usable:
        by_family.setdefault(_METRICS[key].family if key in _METRICS else key, []).append(key)

    scaled: dict[str, float] = {}
    for family, keys in by_family.items():
        total = sum(weights[k] for k in keys)
        if total <= 0:
            continue
        family_weight = _FAMILY_WEIGHTS.get(family, 1.0)
        for k in keys:
            scaled[k] = weights[k] / total * family_weight
    return scaled


def _assess_one(
    embedder,
    queries: list[str],
    references: list[str],
    *,
    n_variants: int,
    max_edits: int,
    seed: int,
    max_refs_for_pairs: int | None,
    noise_profile: NoiseProfile | None = None,
) -> dict[str, float]:
    """Compute all label-free metrics for one embedder on the dataset."""
    q_emb = embedder.encode(queries) if queries else np.empty((0, 0))
    r_emb = embedder.encode(references)
    # the centering metrics take r_float, the rest r_norm
    r_float = np.asarray(r_emb, dtype=np.float64)
    r_norm = metrics_mod.l2_normalize(r_float)

    out: dict[str, float] = {}
    out.update(metrics_mod.reference_separability(r_norm, _normalized=True))
    out.update(metrics_mod.centered_separability(r_float))
    out.update(metrics_mod.intrinsic_dimensionality(r_float))
    out["uniformity_loss"] = metrics_mod.uniformity_loss(r_norm, _normalized=True)

    src, var = generate_positive_pairs(
        references,
        n_variants=n_variants,
        max_edits=max_edits,
        seed=seed,
        max_references=max_refs_for_pairs,
        profile=noise_profile,
    )
    if src:
        src_emb, var_emb = embedder.encode(src), embedder.encode(var)
        out["alignment_loss"] = metrics_mod.alignment_loss(src_emb, var_emb)
        out["positive_pair_rank"] = metrics_mod.positive_pair_rank(src_emb, var_emb)
    else:
        out["alignment_loss"] = float("nan")
        out["positive_pair_rank"] = float("nan")

    if len(queries) > 0:
        q_norm = metrics_mod.l2_normalize(q_emb)
        qr_sims = q_norm @ r_norm.T
        out.update(metrics_mod.retrieval_margin(
            q_norm, r_norm, _normalized=True, _sims=qr_sims))
        out.update(metrics_mod.hubness(q_norm, r_norm, _normalized=True, _sims=qr_sims))
        out["mutual_nn_rate"] = metrics_mod.mutual_nn_rate(
            q_norm, r_norm, _normalized=True, _sims=qr_sims)

    out["embedding_dim"] = float(embedder.dim or 0)
    return out


def _composite_scores(
    assessments: list[ModelAssessment],
    weights: dict[str, float],
    *,
    family_normalize: bool = True,
) -> None:
    """Fill in ``.score`` from a cross-model z-scored blend, in place."""
    valid = [a for a in assessments if not a.error]
    if len(valid) < 2:
        return

    for a in valid:
        a.score = 0.0

    if family_normalize:
        available = set()
        for key in {k for a in valid for k in a.metrics}:
            vals = np.array([a.metrics.get(key, np.nan) for a in valid], dtype=np.float64)
            finite = vals[np.isfinite(vals)]
            if finite.size >= 2 and np.std(finite) > 0:
                available.add(key)
        weights = _family_normalized_weights(weights, sorted(available))

    for key in weights:
        spec = _METRICS.get(key)
        vals = np.array([a.metrics.get(key, np.nan) for a in valid], dtype=np.float64)
        if spec is not None and spec.transform is not None:
            vals = spec.transform(vals)
        finite = np.isfinite(vals)
        if finite.sum() < 2:
            continue

        orientation = spec.orientation if spec is not None else 1
        # worst is the max for lower-is-better metrics, the min for higher-is-better
        worst = np.min(vals[finite]) if orientation > 0 else np.max(vals[finite])
        vals = np.where(finite, vals, worst)

        mu = float(np.mean(vals))
        sigma = float(np.std(vals))
        if sigma == 0:
            continue
        z = (vals - mu) / sigma * orientation * weights[key]
        for a, zi in zip(valid, z):
            a.score += float(zi)


def assess_models(
    queries: Sequence[str],
    references: Sequence[str],
    models: dict[str, ModelSpec],
    *,
    force_cpu: bool = True,
    n_variants: int = 2,
    max_edits: int = 2,
    seed: int = 0,
    max_refs_for_pairs: int | None = 300,
    weights: dict[str, float] | None = None,
    family_normalize: bool = True,
    noise_profile: NoiseProfile | None = None,
    estimate_noise: bool = True,
) -> AssessmentReport:
    """Assess candidate embedding models on a dataset, label-free."""
    queries = [q for q in queries if isinstance(q, str) and q.strip()]
    references = [r for r in references if isinstance(r, str) and r.strip()]
    weights = weights or dict(_DEFAULT_WEIGHTS)

    if noise_profile is None and estimate_noise and queries:
        noise_profile = estimate_noise_profile(queries, references)
        logger.info("Estimated noise profile: %s", noise_profile.describe())

    assessments: list[ModelAssessment] = []
    for name, spec in models.items():
        try:
            embedder = CachingEmbedder(as_embedder(spec, force_cpu=force_cpu, name=name))
            metrics = _assess_one(
                embedder,
                queries,
                references,
                n_variants=n_variants,
                max_edits=max_edits,
                seed=seed,
                max_refs_for_pairs=max_refs_for_pairs,
                noise_profile=noise_profile,
            )
            assessments.append(ModelAssessment(name=name, metrics=metrics))
        except Exception as e:  # surface per-model failures without aborting the batch
            assessments.append(ModelAssessment(name=name, error=str(e)))

    failed = [a for a in assessments if a.error]
    if failed and len(failed) < len(assessments):
        logger.warning(
            "%d of %d models failed and are excluded from the composite: %s",
            len(failed),
            len(assessments),
            "; ".join(f"{a.name}: {a.error}" for a in failed),
        )

    _composite_scores(assessments, weights, family_normalize=family_normalize)

    return AssessmentReport(
        assessments=assessments,
        n_queries=len(queries),
        n_references=len(references),
        config={
            "n_variants": n_variants,
            "max_edits": max_edits,
            "seed": seed,
            "weights": weights,
            "family_normalize": family_normalize,
        },
    )


def assessment_table(*args, **kwargs):
    """Convenience: run :func:`assess_models` and return only the tidy table."""
    return assess_models(*args, **kwargs).to_table()
