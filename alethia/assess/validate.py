"""Check whether the label-free score predicts true model ranking on labeled data."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..embedder import ModelSpec, as_embedder, l2_normalize
from .assessor import _DEFAULT_WEIGHTS, assess_models


@dataclass
class LabeledDataset:
    """A dataset with a known correct answer for each query.

    Attributes:
        name: Identifier for the dataset.
        queries: Query strings.
        references: Candidate reference strings.
        truth: For each query, the correct reference string (must be in ``references``).
    """

    name: str
    queries: List[str]
    references: List[str]
    truth: List[str]

    def __post_init__(self) -> None:
        if not (len(self.queries) == len(self.truth)):
            raise ValueError("queries and truth must have equal length")
        ref_set = set(self.references)
        missing = [t for t in self.truth if t not in ref_set]
        if missing:
            raise ValueError(
                f"{len(missing)} ground-truth answers are not in references "
                f"(e.g. {missing[0]!r})"
            )


@dataclass
class ValidationResult:
    """Outcome of the meta-evaluation."""

    per_model_dataset: List[Dict[str, Any]]
    kendall_tau: float
    spearman_rho: float
    n_datasets: int
    n_models: int
    notes: Dict[str, Any] = field(default_factory=dict)

    def to_table(self):
        """Per (model, dataset) true accuracy and label-free score as a DataFrame."""
        import pandas as pd

        return pd.DataFrame(self.per_model_dataset)


def true_accuracy(
    queries: Sequence[str],
    references: Sequence[str],
    truth: Sequence[str],
    embedder,
) -> Dict[str, float]:
    """Top-1 accuracy and mean reciprocal rank of an embedder against ground truth."""
    queries = list(queries)
    references = list(references)
    if not queries or not references:
        return {"top1": float("nan"), "mrr": float("nan")}

    q = l2_normalize(embedder.encode(queries))
    r = l2_normalize(embedder.encode(references))
    sims = q @ r.T  # (nq, nr)
    order = np.argsort(-sims, axis=1)  # ranked reference indices per query

    ref_index = {ref: i for i, ref in enumerate(references)}
    top1 = 0
    rr_sum = 0.0
    for i, correct in enumerate(truth):
        gold = ref_index[correct]
        rank = int(np.where(order[i] == gold)[0][0])  # 0-based
        if rank == 0:
            top1 += 1
        rr_sum += 1.0 / (rank + 1)
    n = len(queries)
    return {"top1": top1 / n, "mrr": rr_sum / n}


def _kendall_tau(a: Sequence[float], b: Sequence[float]) -> float:
    """Kendall's tau-b between two rankings (no SciPy dependency)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(a)
    if n < 2:
        return float("nan")
    concordant = discordant = ties_a = ties_b = 0
    for i in range(n):
        for j in range(i + 1, n):
            da = a[i] - a[j]
            db = b[i] - b[j]
            if da == 0 and db == 0:
                continue
            if da == 0:
                ties_a += 1
                continue
            if db == 0:
                ties_b += 1
                continue
            if (da > 0) == (db > 0):
                concordant += 1
            else:
                discordant += 1
    denom = np.sqrt(
        (concordant + discordant + ties_a) * (concordant + discordant + ties_b)
    )
    if denom == 0:
        return float("nan")
    return (concordant - discordant) / denom


def _spearman_rho(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman's rho between two sequences (no SciPy dependency)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(a)
    if n < 2:
        return float("nan")

    def ranks(x):
        order = np.argsort(x)
        r = np.empty(n, dtype=float)
        r[order] = np.arange(n, dtype=float)
        return r

    ra, rb = ranks(a), ranks(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    if denom == 0:
        return float("nan")
    return float((ra * rb).sum() / denom)


def validate_assessor(
    datasets: Sequence[LabeledDataset],
    models: Dict[str, ModelSpec],
    *,
    force_cpu: bool = True,
    metric: str = "top1",
    weights: Optional[Dict[str, float]] = None,
    seed: int = 0,
) -> ValidationResult:
    """Check that the label-free composite predicts true model ranking.

    Args:
        datasets: Labeled datasets to evaluate over.
        models: Candidate models (any spec accepted by :func:`as_embedder`).
        force_cpu: Prefer CPU when loading named models.
        metric: True-performance metric to rank against (``"top1"`` or ``"mrr"``).
        weights: Optional composite-weight override.
        seed: Passed to the dirty-variant simulator inside the assessor.

    Returns:
        A :class:`ValidationResult`. ``kendall_tau`` / ``spearman_rho`` correlate the
        label-free predicted ranking with the true ranking, pooled across datasets under
        leave-one-dataset-out (each dataset's models ranked among themselves, then
        correlations pooled over all dataset-internal model pairs).
    """
    if metric not in ("top1", "mrr"):
        raise ValueError("metric must be 'top1' or 'mrr'")
    weights = weights or dict(_DEFAULT_WEIGHTS)

    rows: List[Dict[str, Any]] = []
    embedders = {
        name: as_embedder(spec, force_cpu=force_cpu, name=name)
        for name, spec in models.items()
    }

    for ds in datasets:
        report = assess_models(
            ds.queries,
            ds.references,
            models,
            force_cpu=force_cpu,
            weights=weights,
            seed=seed,
        )
        score_by_model = {a.name: a.score for a in report.assessments if not a.error}
        for name, emb in embedders.items():
            acc = true_accuracy(ds.queries, ds.references, ds.truth, emb)
            rows.append(
                {
                    "dataset": ds.name,
                    "model": name,
                    "true_top1": acc["top1"],
                    "true_mrr": acc["mrr"],
                    "label_free_score": score_by_model.get(name, float("nan")),
                }
            )

    # Correlate predicted vs true ranking *within* each dataset, then pool the per-dataset
    # correlations. This is the leave-one-dataset-out spirit: a model's label-free score on a
    # dataset is compared to its true rank on that same held-out dataset.
    true_key = "true_top1" if metric == "top1" else "true_mrr"
    taus, rhos = [], []
    for ds in datasets:
        sub = [r for r in rows if r["dataset"] == ds.name]
        pred = [r["label_free_score"] for r in sub]
        true = [r[true_key] for r in sub]
        if len(sub) >= 2 and np.all(np.isfinite(pred)) and np.all(np.isfinite(true)):
            taus.append(_kendall_tau(pred, true))
            rhos.append(_spearman_rho(pred, true))

    tau = float(np.nanmean(taus)) if taus else float("nan")
    rho = float(np.nanmean(rhos)) if rhos else float("nan")

    return ValidationResult(
        per_model_dataset=rows,
        kendall_tau=tau,
        spearman_rho=rho,
        n_datasets=len(datasets),
        n_models=len(models),
        notes={"metric": metric, "datasets_with_correlation": len(taus)},
    )
