"""Label-free embedding quality metrics for entity matching."""


import numpy as np

from ..embedder import l2_normalize, rank_key, top_k_stable


def _ensure_normalized(arr: np.ndarray, *, normalized: bool = False) -> np.ndarray:
    """Return a float64 normalized matrix, trusting an internal normalized input."""
    if normalized:
        result = np.asarray(arr)
        if result.dtype != np.float64:
            raise TypeError("pre-normalized metric inputs must have dtype float64")
        return result
    return l2_normalize(arr)




#: 4096 x n float64 stays under a GB
_BLOCK_ROWS = 4096
#: sorting and partitioning copy their input, so blocks stay near 20 MB
_RETRIEVAL_BLOCK_ROWS = 128

def _nearest_other_similarity(emb: np.ndarray, block_rows: int = _BLOCK_ROWS) -> np.ndarray:
    """Cosine similarity from each row to its nearest *other* row."""
    n = emb.shape[0]
    out = np.empty(n, dtype=np.float64)
    for start in range(0, n, block_rows):
        stop = min(start + block_rows, n)
        block = emb[start:stop] @ emb.T  # (rows, n)
        block[np.arange(stop - start), np.arange(start, stop)] = -np.inf
        out[start:stop] = block.max(axis=1)
    return out


def reference_separability(
    ref_emb: np.ndarray, confusion_threshold: float = 0.9, *, _normalized: bool = False
) -> dict[str, float]:
    """Nearest-other-reference cosine statistics: mean, median, and confusability rate.

    Lower is better on all three. Raw cosines, so they partly reflect anisotropy
    (Ethayarajh 2019); see :func:`centered_separability` for the corrected view.
    """
    ref_emb = _ensure_normalized(ref_emb, normalized=_normalized)
    n = ref_emb.shape[0]
    if n < 2:
        return {
            "mean_nn_similarity": 0.0,
            "median_nn_similarity": 0.0,
            "confusability_rate": 0.0,
        }
    nn_sim = _nearest_other_similarity(ref_emb)
    return {
        "mean_nn_similarity": float(np.mean(nn_sim)),
        "median_nn_similarity": float(np.median(nn_sim)),
        "confusability_rate": float(np.mean(nn_sim > confusion_threshold)),
    }


def centered_separability(ref_emb: np.ndarray) -> dict[str, float]:
    """Anisotropy-corrected separability: nearest-neighbour margin after mean removal.

    Returns ``centered_nn_similarity`` (lower better) and ``nn_margin_z``, how many
    standard deviations the top neighbour stands above a typical one (scale-free).

    References:
        Ethayarajh (2019), doi:10.48550/arXiv.1909.00512. Su et al. (2021),
        doi:10.48550/arXiv.2103.15316.
    """
    emb = np.asarray(ref_emb, dtype=np.float64)
    n = emb.shape[0]
    if n < 3:
        return {"centered_nn_similarity": 0.0, "nn_margin_z": 0.0}

    centered = l2_normalize(emb - emb.mean(axis=0, keepdims=True))

    # one blocked pass produces all three accumulators
    nn_sim = np.empty(n, dtype=np.float64)
    sums = np.empty(n, dtype=np.float64)
    sqsums = np.empty(n, dtype=np.float64)
    for start in range(0, n, _BLOCK_ROWS):
        stop = min(start + _BLOCK_ROWS, n)
        block = centered[start:stop] @ centered.T
        rows, cols = np.arange(stop - start), np.arange(start, stop)
        # self-similarity is exactly 1.0 on normalized rows
        sums[start:stop] = block.sum(axis=1) - 1.0
        # avoids materializing block**2
        sqsums[start:stop] = np.einsum("ij,ij->i", block, block) - 1.0
        block[rows, cols] = -np.inf
        nn_sim[start:stop] = block.max(axis=1)

    m = n - 1
    mu = sums / m
    var = np.maximum(0.0, sqsums / m - mu**2)
    sd = np.sqrt(var)
    sd[sd == 0] = np.nan  # degenerate rows drop out of the nanmean below
    margin_z = (nn_sim - mu) / sd

    return {
        "centered_nn_similarity": float(np.mean(nn_sim)),
        "nn_margin_z": float(np.nanmean(margin_z)),
    }


def intrinsic_dimensionality(emb: np.ndarray) -> dict[str, float]:
    """Effective dimensionality as a PCA participation ratio.

    Returns ``participation_ratio`` and ``normalized_pr``, its ratio to the ambient dim.
    """
    emb = np.asarray(emb, dtype=np.float64)
    if emb.shape[0] < 2:
        return {"participation_ratio": 0.0, "normalized_pr": 0.0}
    centered = emb - emb.mean(axis=0, keepdims=True)
    # the d x d form has the same nonzero eigenvalues as the n x n one
    lam = np.linalg.eigvalsh(centered.T @ centered)
    lam = np.maximum(lam, 0.0)
    total = np.sum(lam)
    if total <= 0:
        return {"participation_ratio": 0.0, "normalized_pr": 0.0}
    pr = (total**2) / np.sum(lam**2)
    return {
        "participation_ratio": float(pr),
        "normalized_pr": float(pr / emb.shape[1]),
    }




def alignment_loss(
    src_emb: np.ndarray, var_emb: np.ndarray, alpha: float = 2.0
) -> float:
    """Alignment: mean distance^alpha between positive pairs (lower = better)."""
    if src_emb.shape[0] == 0:
        return float("nan")
    a = l2_normalize(src_emb)
    b = l2_normalize(var_emb)
    d = np.linalg.norm(a - b, axis=1)
    return float(np.mean(d**alpha))


def positive_pair_rank(
    src_emb: np.ndarray, var_emb: np.ndarray, block_rows: int = _BLOCK_ROWS
) -> float:
    """Mean percentile rank of each positive variant among all variants, ties mid-ranked."""
    if src_emb.shape[0] == 0 or var_emb.shape[0] == 0:
        return float("nan")
    if src_emb.shape[0] != var_emb.shape[0]:
        raise ValueError("src_emb and var_emb must have the same number of rows")

    src = l2_normalize(src_emb)
    var = l2_normalize(var_emb)
    n = src.shape[0]
    ranks = np.empty(n, dtype=np.float64)
    for start in range(0, n, block_rows):
        stop = min(start + block_rows, n)
        sims = src[start:stop] @ var.T
        positive = sims[np.arange(stop - start), np.arange(start, stop)]
        # rounded, so ties compare equal on every BLAS build
        sims = rank_key(sims)
        positive = rank_key(positive)
        lower = np.sum(sims < positive[:, None], axis=1)
        equal = np.sum(sims == positive[:, None], axis=1)
        ranks[start:stop] = (lower + 0.5 * equal) / n
    return float(np.mean(ranks))


def uniformity_loss(
    emb: np.ndarray, t: float = 2.0, max_points: int = 2000, *, _normalized: bool = False
) -> float:
    """Uniformity ``log E exp(-t * ||x-y||^2)`` over pairs; more negative is better.

    After Wang & Isola (2020), doi:10.48550/arXiv.2005.10242.
    """
    n = emb.shape[0]
    if n < 2:
        return float("nan")
    if n > max_points:
        idx = np.linspace(0, n - 1, max_points).astype(int)
        emb = emb[idx]
    emb = _ensure_normalized(emb, normalized=_normalized)
    # ||x - y||^2 = 2 - 2 x.y on unit rows, so no distance array is built
    gram = emb @ emb.T
    total = 0.0
    pairs = 0
    for row in range(emb.shape[0] - 1):
        pdist = np.maximum(0.0, 2.0 - 2.0 * gram[row, row + 1 :])
        total += float(np.exp(-t * pdist).sum())
        pairs += pdist.size
    return float(np.log(total / pairs))




def _query_ref_sims(
    query_emb: np.ndarray, ref_emb: np.ndarray, *, normalized: bool = False
) -> np.ndarray:
    return (_ensure_normalized(query_emb, normalized=normalized)
            @ _ensure_normalized(ref_emb, normalized=normalized).T)


def retrieval_margin(
    query_emb: np.ndarray, ref_emb: np.ndarray, low_margin_z: float = 1.0, *,
    _normalized: bool = False, _sims: np.ndarray | None = None,
) -> dict[str, float]:
    """Top-1 vs top-2 margin per query, in raw cosine and in standard deviations.

    ``mean_margin`` / ``low_margin_rate`` are absolute cosines and so confounded by
    anisotropy; the ``_z`` variants divide by the per-query spread and compare fairly
    across models, which is why the composite weights those.
    """
    sims = (_query_ref_sims(query_emb, ref_emb, normalized=_normalized)
            if _sims is None else _sims)
    if sims.shape[1] < 2:
        return {
            "mean_margin": 0.0,
            "low_margin_rate": 1.0,
            "mean_margin_z": 0.0,
            "low_margin_z_rate": 1.0,
        }
    margin = np.empty(sims.shape[0], dtype=np.float64)
    sd = np.empty(sims.shape[0], dtype=np.float64)
    for start in range(0, sims.shape[0], _RETRIEVAL_BLOCK_ROWS):
        stop = min(start + _RETRIEVAL_BLOCK_ROWS, sims.shape[0])
        block = sims[start:stop]
        part = np.partition(block, -2, axis=1)
        margin[start:stop] = part[:, -1] - part[:, -2]
        sd[start:stop] = block.std(axis=1)
    sd = np.where(sd == 0, np.nan, sd)
    margin_z = margin / sd

    return {
        "mean_margin": float(np.mean(margin)),
        "low_margin_rate": float(np.mean(margin < 0.02)),
        "mean_margin_z": float(np.nanmean(margin_z)),
        "low_margin_z_rate": float(
            np.nanmean(np.where(np.isnan(margin_z), np.nan, margin_z < low_margin_z))
        ),
    }


def hubness(
    query_emb: np.ndarray, ref_emb: np.ndarray, k: int = 5, *,
    _normalized: bool = False, _sims: np.ndarray | None = None,
) -> dict[str, float]:
    """Skewness of the k-occurrence distribution; high means pathological hubs.

    After Radovanovic et al. (2010), JMLR 11:2487-2531.
    """
    q = _ensure_normalized(query_emb, normalized=_normalized)
    r = _ensure_normalized(ref_emb, normalized=_normalized)
    n_ref = r.shape[0]
    k = min(k, n_ref)
    if k < 1:
        return {"hubness_skew": 0.0}
    counts = np.zeros(n_ref, dtype=np.float64)
    for start in range(0, q.shape[0], _RETRIEVAL_BLOCK_ROWS):
        stop = min(start + _RETRIEVAL_BLOCK_ROWS, q.shape[0])
        sims = q[start:stop] @ r.T if _sims is None else _sims[start:stop]
        topk_idx = top_k_stable(sims, k)
        counts += np.bincount(topk_idx.ravel(), minlength=n_ref)
    mu, sigma = counts.mean(), counts.std()
    if sigma == 0:
        return {"hubness_skew": 0.0}
    skew = float(np.mean(((counts - mu) / sigma) ** 3))
    return {"hubness_skew": skew}


def mutual_nn_rate(
    query_emb: np.ndarray, ref_emb: np.ndarray, k: int = 5, *,
    _normalized: bool = False, _sims: np.ndarray | None = None,
) -> float:
    """Fraction of query->ref top-1 assignments that are reciprocated within ref top-k."""
    q = _ensure_normalized(query_emb, normalized=_normalized)
    r = _ensure_normalized(ref_emb, normalized=_normalized)
    nq, nr = q.shape[0], r.shape[0]
    if nq == 0 or nr == 0:
        return float("nan")
    k = min(k, nq)

    # blocked so the (nq, nr) matrix is never held whole
    q_top1 = np.empty(nq, dtype=np.intp)
    for start in range(0, nq, _BLOCK_ROWS):
        stop = min(start + _BLOCK_ROWS, nq)
        block = q[start:stop] @ r.T if _sims is None else _sims[start:stop]
        q_top1[start:stop] = np.argmax(rank_key(block), axis=1)

    # only distinct chosen references need a query row, avoiding an (nq, nq) gather
    chosen_refs, inverse = np.unique(q_top1, return_inverse=True)
    topk = np.empty((len(chosen_refs), k), dtype=np.intp)
    # the _sims branch gathers columns, copying (nq, block)
    for start in range(0, len(chosen_refs), _RETRIEVAL_BLOCK_ROWS):
        stop = min(start + _RETRIEVAL_BLOCK_ROWS, len(chosen_refs))
        block = (r[chosen_refs[start:stop]] @ q.T if _sims is None
                else _sims[:, chosen_refs[start:stop]].T)
        topk[start:stop] = top_k_stable(block, k)

    # topk[inverse] is (nq, k): each query's chosen reference's top-k queries
    return float((topk[inverse] == np.arange(nq)[:, None]).any(axis=1).mean())
