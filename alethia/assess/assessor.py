"""Orchestrates per-model label-free metrics into a composite score and report."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..embedder import ModelSpec, as_embedder
from . import metrics as metrics_mod
from .simulate import generate_positive_pairs


@dataclass
class ModelAssessment:
    """Label-free assessment of a single model on one dataset."""

    name: str
    metrics: Dict[str, float] = field(default_factory=dict)
    score: float = float("nan")
    error: Optional[str] = None

    def as_row(self) -> Dict[str, Any]:
        row: Dict[str, Any] = {"model": self.name, "score": self.score}
        row.update(self.metrics)
        if self.error:
            row["error"] = self.error
        return row


@dataclass
class AssessmentReport:
    """Assessment of several models on one dataset."""

    assessments: List[ModelAssessment]
    n_queries: int
    n_references: int
    config: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_comparative(self) -> bool:
        """True when at least two models were assessed, so the composite score is meaningful."""
        return sum(not a.error for a in self.assessments) >= 2

    @property
    def best(self) -> Optional[ModelAssessment]:
        """Highest-scoring model, or the sole assessed model when only one was given.

        Returns ``None`` only if no model could be assessed. With a single model the
        composite score is ``NaN`` (not comparative) but that model is still returned.
        """
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
        self, path: Optional[str] = None, title: str = "Alethia Model Assessment"
    ) -> str:
        """Render a professional, self-contained HTML report (see :mod:`.report`)."""
        from .report import render_html

        return render_html(self, path=path, title=title)


# Orientation: +1 if higher-is-better, -1 if lower-is-better. Used to build the composite.
_METRIC_ORIENTATION = {
    "mean_nn_similarity": -1,  # references too close = bad
    "confusability_rate": -1,
    "normalized_pr": +1,  # richer structure = good
    "alignment_loss": -1,  # positives should stay close
    "uniformity_loss": -1,  # more negative = more uniform = good
    "mean_margin": +1,  # confident top-1 = good
    "low_margin_rate": -1,
    "hubness_skew": -1,  # high hubness = bad (use abs)
    "mutual_nn_rate": +1,  # reciprocity = good
}

# Default weights for the composite (documented; sums need not be 1).
_DEFAULT_WEIGHTS = {
    "mean_nn_similarity": 1.0,
    "confusability_rate": 1.0,
    "normalized_pr": 0.5,
    "alignment_loss": 1.5,
    "uniformity_loss": 1.0,
    "mean_margin": 1.5,
    "low_margin_rate": 0.5,
    "hubness_skew": 0.75,
    "mutual_nn_rate": 1.0,
}


def _assess_one(
    embedder,
    queries: List[str],
    references: List[str],
    *,
    n_variants: int,
    max_edits: int,
    seed: int,
    max_refs_for_pairs: Optional[int],
) -> Dict[str, float]:
    """Compute all label-free metrics for one embedder on the dataset."""
    q_emb = embedder.encode(queries) if queries else np.empty((0, 0))
    r_emb = embedder.encode(references)

    out: Dict[str, float] = {}
    out.update(metrics_mod.reference_separability(r_emb))
    out.update(metrics_mod.intrinsic_dimensionality(r_emb))
    out["uniformity_loss"] = metrics_mod.uniformity_loss(r_emb)

    # Alignment via synthesized positive pairs from the references.
    src, var = generate_positive_pairs(
        references,
        n_variants=n_variants,
        max_edits=max_edits,
        seed=seed,
        max_references=max_refs_for_pairs,
    )
    if src:
        out["alignment_loss"] = metrics_mod.alignment_loss(
            embedder.encode(src), embedder.encode(var)
        )
    else:
        out["alignment_loss"] = float("nan")

    if len(queries) > 0:
        out.update(metrics_mod.retrieval_margin(q_emb, r_emb))
        out.update(metrics_mod.hubness(q_emb, r_emb))
        out["mutual_nn_rate"] = metrics_mod.mutual_nn_rate(q_emb, r_emb)

    out["embedding_dim"] = float(embedder.dim or 0)
    return out


def _composite_scores(
    assessments: List[ModelAssessment], weights: Dict[str, float]
) -> None:
    """Fill in ``.score`` for each assessment via a cross-model z-scored blend (in place).

    The composite ranks models *relative to each other*, so it needs at least two models.
    With a single model the score is left as ``NaN`` (its component metrics are still
    reported); see :attr:`AssessmentReport.is_comparative`.
    """
    valid = [a for a in assessments if not a.error]
    if len(valid) < 2:
        return

    for a in valid:
        a.score = 0.0

    # Metrics absent from the models yield all-NaN vectors and are skipped by the finite
    # check below, so iterating every weight key is equivalent to pre-filtering.
    for key in weights:
        vals = np.array([a.metrics.get(key, np.nan) for a in valid], dtype=np.float64)
        if key == "hubness_skew":
            vals = np.abs(vals)
        finite = np.isfinite(vals)
        if finite.sum() < 2:
            continue
        mu = np.nanmean(vals[finite])
        sigma = np.nanstd(vals[finite])
        if sigma == 0:
            continue
        z = (vals - mu) / sigma
        z = z * _METRIC_ORIENTATION.get(key, 1) * weights[key]
        for a, zi in zip(valid, z):
            if np.isfinite(zi):
                a.score += float(zi)


def assess_models(
    queries: Sequence[str],
    references: Sequence[str],
    models: Dict[str, ModelSpec],
    *,
    force_cpu: bool = True,
    n_variants: int = 2,
    max_edits: int = 2,
    seed: int = 0,
    max_refs_for_pairs: Optional[int] = 300,
    weights: Optional[Dict[str, float]] = None,
) -> AssessmentReport:
    """Assess candidate embedding models on a dataset, label-free.

    Args:
        queries: Dirty/query strings (may be empty to assess reference structure only).
        references: Canonical reference strings.
        models: Mapping of display name -> model spec. Each spec is anything
            :func:`alethia.embedder.as_embedder` accepts: a callable ``embed_fn``, a model
            name string, or a loaded model object.
        force_cpu: Prefer CPU when loading named models.
        n_variants, max_edits, seed: Dirty-variant simulator settings (alignment signal).
        max_refs_for_pairs: Cap references used for positive-pair synthesis (speed).
        weights: Optional override of composite metric weights.

    Returns:
        An :class:`AssessmentReport`.

    References:
        Wang & Isola (2020), Understanding contrastive representation learning through
        alignment and uniformity on the hypersphere. doi:10.48550/arXiv.2005.10242.
        Radovanović et al. (2010), Hubs in space: popular nearest neighbors in
        high-dimensional data. JMLR 11:2487-2531.
        https://www.jmlr.org/papers/v11/radovanovic10a.html.
    """
    queries = [q for q in queries if isinstance(q, str) and q.strip()]
    references = [r for r in references if isinstance(r, str) and r.strip()]
    weights = weights or dict(_DEFAULT_WEIGHTS)

    assessments: List[ModelAssessment] = []
    for name, spec in models.items():
        try:
            embedder = as_embedder(spec, force_cpu=force_cpu, name=name)
            metrics = _assess_one(
                embedder,
                queries,
                references,
                n_variants=n_variants,
                max_edits=max_edits,
                seed=seed,
                max_refs_for_pairs=max_refs_for_pairs,
            )
            assessments.append(ModelAssessment(name=name, metrics=metrics))
        except Exception as e:  # surface per-model failures without aborting the batch
            assessments.append(ModelAssessment(name=name, error=str(e)))

    _composite_scores(assessments, weights)

    return AssessmentReport(
        assessments=assessments,
        n_queries=len(queries),
        n_references=len(references),
        config={
            "n_variants": n_variants,
            "max_edits": max_edits,
            "seed": seed,
            "weights": weights,
        },
    )


def assessment_table(*args, **kwargs):
    """Convenience: run :func:`assess_models` and return only the tidy table."""
    return assess_models(*args, **kwargs).to_table()
