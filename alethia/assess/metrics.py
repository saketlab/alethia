"""Label-free embedding quality metrics for entity matching."""

from typing import Dict

import numpy as np

from ..embedder import l2_normalize


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    """Float64 L2 normalization for metric precision (delegates to the canonical helper)."""
    return l2_normalize(arr, dtype=np.float64)


# --------------------------------------------------------------------------------------
# Reference-set separability (§4.1)
# --------------------------------------------------------------------------------------


def reference_separability(
    ref_emb: np.ndarray, confusion_threshold: float = 0.9
) -> Dict[str, float]:
    """Measure how well distinct references are separated in embedding space.

    Returns:
        mean_nn_similarity: mean cosine to nearest *other* reference (lower = better).
        median_nn_similarity: median of the above.
        confusability_rate: fraction of refs whose nearest other ref exceeds the
            confusion threshold (lower = better).
    """
    ref_emb = _l2_normalize(ref_emb)
    n = ref_emb.shape[0]
    if n < 2:
        return {
            "mean_nn_similarity": 0.0,
            "median_nn_similarity": 0.0,
            "confusability_rate": 0.0,
        }
    sims = ref_emb @ ref_emb.T
    np.fill_diagonal(sims, -np.inf)
    nn_sim = np.max(sims, axis=1)
    return {
        "mean_nn_similarity": float(np.mean(nn_sim)),
        "median_nn_similarity": float(np.median(nn_sim)),
        "confusability_rate": float(np.mean(nn_sim > confusion_threshold)),
    }


def intrinsic_dimensionality(emb: np.ndarray) -> Dict[str, float]:
    """Estimate effective dimensionality of an embedding set.

    Returns:
        participation_ratio: (Σλ)² / Σλ² over PCA eigenvalues; higher = richer structure.
        normalized_pr: participation_ratio / ambient_dim, in (0, 1].
    """
    emb = np.asarray(emb, dtype=np.float64)
    if emb.shape[0] < 2:
        return {"participation_ratio": 0.0, "normalized_pr": 0.0}
    centered = emb - emb.mean(axis=0, keepdims=True)
    # Eigenvalues of covariance via SVD of centered data.
    s = np.linalg.svd(centered, compute_uv=False)
    lam = s**2
    total = np.sum(lam)
    if total <= 0:
        return {"participation_ratio": 0.0, "normalized_pr": 0.0}
    pr = (total**2) / np.sum(lam**2)
    return {
        "participation_ratio": float(pr),
        "normalized_pr": float(pr / emb.shape[1]),
    }


# --------------------------------------------------------------------------------------
# Alignment / uniformity (§4.2)
# --------------------------------------------------------------------------------------


def alignment_loss(
    src_emb: np.ndarray, var_emb: np.ndarray, alpha: float = 2.0
) -> float:
    """Alignment: mean distance^alpha between positive pairs (lower = better).

    ``src_emb[i]`` and ``var_emb[i]`` are a (clean, dirty-variant) positive pair.
    """
    if src_emb.shape[0] == 0:
        return float("nan")
    a = _l2_normalize(src_emb)
    b = _l2_normalize(var_emb)
    d = np.linalg.norm(a - b, axis=1)
    return float(np.mean(d**alpha))


def uniformity_loss(emb: np.ndarray, t: float = 2.0, max_points: int = 2000) -> float:
    """Uniformity: log E exp(-t * ||x-y||^2) over pairs.

    More negative = more uniformly spread on the hypersphere (better). Subsamples for
    tractability when there are many points. After Wang & Isola (2020),
    doi:10.48550/arXiv.2005.10242.
    """
    emb = _l2_normalize(emb)
    n = emb.shape[0]
    if n < 2:
        return float("nan")
    if n > max_points:
        # Deterministic stride subsample.
        idx = np.linspace(0, n - 1, max_points).astype(int)
        emb = emb[idx]
    # Squared pairwise distances via the Gram matrix: for L2-normalized rows,
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x.y = 2 - 2 x.y. This avoids materializing
    # the (n, n, d) difference tensor (which OOMs for real embedding dims).
    gram = emb @ emb.T
    iu = np.triu_indices(emb.shape[0], k=1)
    pdist = np.maximum(0.0, 2.0 - 2.0 * gram[iu])  # clamp tiny negatives from fp error
    return float(np.log(np.mean(np.exp(-t * pdist))))


# --------------------------------------------------------------------------------------
# Retrieval self-consistency (§4.3)
# --------------------------------------------------------------------------------------


def _query_ref_sims(query_emb: np.ndarray, ref_emb: np.ndarray) -> np.ndarray:
    return _l2_normalize(query_emb) @ _l2_normalize(ref_emb).T


def retrieval_margin(query_emb: np.ndarray, ref_emb: np.ndarray) -> Dict[str, float]:
    """Top-1 vs top-2 cosine margin per query (higher = more confident/better)."""
    sims = _query_ref_sims(query_emb, ref_emb)
    if sims.shape[1] < 2:
        return {"mean_margin": 0.0, "low_margin_rate": 1.0}
    part = np.partition(sims, -2, axis=1)
    top1 = part[:, -1]
    top2 = part[:, -2]
    margin = top1 - top2
    return {
        "mean_margin": float(np.mean(margin)),
        "low_margin_rate": float(np.mean(margin < 0.02)),
    }


def hubness(query_emb: np.ndarray, ref_emb: np.ndarray, k: int = 5) -> Dict[str, float]:
    """Skewness of the k-occurrence distribution (lower = better; high = pathological).

    Hubness: some references become top-k neighbours of disproportionately many queries,
    which degrades retrieval. After Radovanović et al. (2010), JMLR 11:2487-2531,
    https://www.jmlr.org/papers/v11/radovanovic10a.html.
    """
    sims = _query_ref_sims(query_emb, ref_emb)
    n_ref = sims.shape[1]
    k = min(k, n_ref)
    if k < 1:
        return {"hubness_skew": 0.0}
    topk_idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    counts = np.bincount(topk_idx.ravel(), minlength=n_ref).astype(np.float64)
    mu, sigma = counts.mean(), counts.std()
    if sigma == 0:
        return {"hubness_skew": 0.0}
    skew = float(np.mean(((counts - mu) / sigma) ** 3))
    return {"hubness_skew": skew}


def mutual_nn_rate(query_emb: np.ndarray, ref_emb: np.ndarray, k: int = 5) -> float:
    """Fraction of query->ref top-1 assignments that are reciprocated within ref top-k.

    Higher = more stable, trustworthy matches.
    """
    q = _l2_normalize(query_emb)
    r = _l2_normalize(ref_emb)
    qr = q @ r.T  # (nq, nr)
    nq, nr = qr.shape
    if nq == 0 or nr == 0:
        return float("nan")
    q_top1 = np.argmax(qr, axis=1)  # best ref j for each query i
    rq = qr.T  # (nr, nq); ref->query similarities are the transpose
    k = min(k, nq)
    # For each query i with chosen ref j, is i among ref j's top-k queries?
    # Vectorized: gather the rows rq[q_top1] (one per query), take their top-k query
    # indices, and test membership of each query's own index i.
    chosen_rows = rq[q_top1]  # (nq, nq)
    topk = np.argpartition(-chosen_rows, kth=k - 1, axis=1)[:, :k]  # (nq, k)
    own = np.arange(nq)[:, None]
    reciprocated = np.any(topk == own, axis=1).sum()
    return float(reciprocated / nq)
