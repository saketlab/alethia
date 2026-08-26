"""Check whether the label-free score predicts true model ranking on labeled data."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..embedder import CachingEmbedder, ModelSpec, as_embedder, l2_normalize, rank_key
from .assessor import assess_models


@dataclass
class LabeledDataset:
    """A dataset with a known correct answer for each query.

    ``truth[i]`` is the correct reference, or ``None`` for an open-world "NIL" query with no
    correct answer. Scoring a dataset with NIL queries needs an abstention threshold; see
    :func:`true_accuracy`.
    """

    name: str
    queries: list[str]
    references: list[str]
    truth: list[str | None]

    def __post_init__(self) -> None:
        if not (len(self.queries) == len(self.truth)):
            raise ValueError("queries and truth must have equal length")
        ref_set = set(self.references)
        missing = [t for t in self.truth if t is not None and t not in ref_set]
        if missing:
            raise ValueError(
                f"{len(missing)} ground-truth answers are not in references "
                f"(e.g. {missing[0]!r})"
            )

    @property
    def has_nil(self) -> bool:
        """True when some query has no correct reference (open-world evaluation)."""
        return any(t is None for t in self.truth)

    @property
    def nil_rate(self) -> float:
        """Fraction of queries with no correct answer."""
        return (sum(t is None for t in self.truth) / len(self.truth)) if self.truth else 0.0


@dataclass
class ValidationResult:
    """Outcome of the meta-evaluation."""

    per_model_dataset: list[dict[str, Any]]
    kendall_tau: float
    spearman_rho: float
    n_datasets: int
    n_models: int
    notes: dict[str, Any] = field(default_factory=dict)

    def to_table(self):
        """Per (model, dataset) true accuracy and label-free score as a DataFrame."""
        import pandas as pd

        return pd.DataFrame(self.per_model_dataset)


def true_accuracy(
    queries: Sequence[str],
    references: Sequence[str],
    truth: Sequence[str | None],
    embedder,
    nil_threshold: float | None = None,
) -> dict[str, float]:
    """Top-1 accuracy and MRR of an embedder against ground truth."""
    queries = list(queries)
    references = list(references)
    truth = list(truth)
    if not queries or not references:
        return {"top1": float("nan"), "mrr": float("nan")}

    has_nil = any(t is None for t in truth)
    if has_nil and nil_threshold is None:
        raise ValueError(
            "truth contains NIL (None) entries, so nil_threshold is required: without it "
            "there is no rule by which the model can decline to answer."
        )

    q = l2_normalize(embedder.encode(queries))
    r = l2_normalize(embedder.encode(references))
    sims = q @ r.T
    best_sim = sims.max(axis=1)

    ref_index = {ref: i for i, ref in enumerate(references)}
    # rank without sorting: those scoring above, plus ties at a lower index
    key = rank_key(sims)
    gold_idx = np.array(
        [ref_index[t] if t is not None else 0 for t in truth], dtype=np.intp
    )
    gold_key = key[np.arange(len(queries)), gold_idx][:, None]
    ranks = (key > gold_key).sum(axis=1) + (
        (key == gold_key) & (np.arange(len(references)) < gold_idx[:, None])
    ).sum(axis=1)
    top1 = 0
    rr_sum = 0.0
    n_answerable = 0
    answerable_hits = 0
    nil_total = 0
    nil_hits = 0

    for i, correct in enumerate(truth):
        abstained = has_nil and best_sim[i] < nil_threshold
        if correct is None:
            nil_total += 1
            if abstained:
                top1 += 1
                nil_hits += 1
            continue
        n_answerable += 1
        rank = int(ranks[i])
        if rank == 0 and not abstained:
            top1 += 1
            answerable_hits += 1
        if not abstained:
            rr_sum += 1.0 / (rank + 1)

    n = len(queries)
    out = {"top1": top1 / n,
           "mrr": (rr_sum / n_answerable) if n_answerable else float("nan")}
    if has_nil:
        out["nil_recall"] = nil_hits / nil_total if nil_total else float("nan")
        out["answerable_top1"] = (answerable_hits / n_answerable
                                  if n_answerable else float("nan"))
    return out


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
        # mid-ranks for ties, as Spearman is defined; distinct ordinal ranks would
        # depend on the sort's tie order and disagree with scipy and R
        order = np.argsort(x, kind="stable")
        r = np.empty(n, dtype=float)
        r[order] = np.arange(n, dtype=float)
        sx = x[order]
        start = 0
        for stop in range(1, n + 1):
            if stop == n or sx[stop] != sx[start]:
                if stop - start > 1:
                    r[order[start:stop]] = (start + stop - 1) / 2.0
                start = stop
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
    models: dict[str, ModelSpec],
    *,
    force_cpu: bool = True,
    metric: str = "top1",
    weights: dict[str, float] | None = None,
    seed: int = 0,
    nil_threshold: float | None = None,
) -> ValidationResult:
    """Check that the label-free composite predicts true model ranking."""
    if metric not in ("top1", "mrr"):
        raise ValueError("metric must be 'top1' or 'mrr'")

    rows: list[dict[str, Any]] = []
    embedders = {
        name: CachingEmbedder(as_embedder(spec, force_cpu=force_cpu, name=name))
        for name, spec in models.items()
    }

    for ds in datasets:
        report = assess_models(
            ds.queries,
            ds.references,
            embedders,
            force_cpu=force_cpu,
            weights=weights,
            seed=seed,
        )
        score_by_model = {a.name: a.score for a in report.assessments if not a.error}
        for name, emb in embedders.items():
            acc = true_accuracy(ds.queries, ds.references, ds.truth, emb,
                                nil_threshold=nil_threshold)
            rows.append(
                {
                    "dataset": ds.name,
                    "model": name,
                    "true_top1": acc["top1"],
                    "true_mrr": acc["mrr"],
                    "label_free_score": score_by_model.get(name, float("nan")),
                }
            )
        for emb in embedders.values():
            emb.clear()

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
