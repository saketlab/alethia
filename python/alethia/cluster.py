"""Embedding-based entity clustering via mutual nearest neighbours.

Merging everything above a cosine threshold is single-linkage: one hub entity moderately
similar to many others chains them all into one giant cluster, which is exactly the wrong
failure mode for deduplicating messy entity names. Edges here need *mutual* nearest
neighbourhood above a floor, and each carries a confidence folding in the retrieval
margin, so clusters are auditable connected components.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .embedder import Embedder, as_embedder, l2_normalize, rank_key, top_k_stable

__all__ = ["Edge", "ClusterResult", "mutual_nn_edges", "cluster_entities"]


@dataclass
class Edge:
    """A candidate merge between two entities, with quantified confidence."""

    i: int
    j: int
    cosine: float
    margin_i: float
    margin_j: float
    mutual: bool
    confidence: float


@dataclass
class ClusterResult:
    """Result of clustering: per-entity cluster ids plus the edges and metadata."""

    entities: list[str]
    labels: list[int]
    edges: list[Edge]
    canonical: dict[int, str]
    embedder_name: str

    def n_clusters(self) -> int:
        return len(set(self.labels))

    def clusters(self) -> dict[int, list[str]]:
        out: dict[int, list[str]] = {}
        for ent, lab in zip(self.entities, self.labels):
            out.setdefault(lab, []).append(ent)
        return out

    def to_records(self) -> list[dict]:
        """One row per entity: entity, cluster, canonical name."""
        return [
            {"entity": e, "cluster": lab, "canonical": self.canonical[lab]}
            for e, lab in zip(self.entities, self.labels)
        ]

    def edge_records(self) -> list[dict]:
        """One row per candidate edge, sorted by confidence (for review/audit)."""
        rows = [
            {
                "entity_a": self.entities[e.i],
                "entity_b": self.entities[e.j],
                "cosine": round(e.cosine, 4),
                "margin": round(min(e.margin_i, e.margin_j), 4),
                "mutual": e.mutual,
                "confidence": round(e.confidence, 4),
            }
            for e in self.edges
        ]
        rows.sort(key=lambda r: -r["confidence"])
        return rows


def _topk(masked_sims: np.ndarray, k: int) -> np.ndarray:
    """Top-k indices per row of an already diagonal-masked matrix, descending."""
    n = masked_sims.shape[0]
    k = max(1, min(k, n - 1))
    return top_k_stable(masked_sims, k)


def mutual_nn_edges(
    emb: np.ndarray,
    *,
    floor: float = 0.80,
    k: int = 5,
    require_mutual: bool = True,
) -> list[Edge]:
    """Build candidate merge edges from mutual nearest neighbours."""
    # at float32 two equally-similar entities can round apart
    x = l2_normalize(np.asarray(emb, dtype=np.float64))
    with np.errstate(invalid="ignore"):
        sims = x @ x.T
    if sims.size and not np.isfinite(sims).all():
        raise ValueError("Non-finite similarity matrix; embeddings contain NaN or inf.")
    n = x.shape[0]
    if n < 2:
        return []

    # every value read below is off-diagonal, so one masked copy serves both uses
    np.fill_diagonal(sims, -np.inf)

    # set iteration order would otherwise leak into confidence ties
    ranked = [[int(j) for j in row] for row in _topk(sims, k)]
    neighbours = [set(row) for row in ranked]

    if n > 2:
        part = np.partition(sims, -2, axis=1)
        margin = part[:, -1] - part[:, -2]
    else:
        # one real off-diagonal per row, so there is no runner-up to measure against
        margin = np.zeros(n, dtype=np.float64)

    seen = set()
    edges: list[Edge] = []
    keyed = rank_key(sims)
    floor_key = float(rank_key(np.asarray(floor)))
    for i in range(n):
        for j in ranked[i]:
            c = float(sims[i, j])
            if float(keyed[i, j]) < floor_key:
                continue
            mutual = i in neighbours[j]
            if require_mutual and not mutual:
                continue
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            ma, mb = float(margin[a]), float(margin[b])
            conf = c * (1.0 + min(ma, mb))
            edges.append(Edge(a, b, c, ma, mb, mutual, conf))
    # total order, so edge_records() is stable
    edges.sort(key=lambda e: (-e.confidence, e.i, e.j))
    return edges


def _connected_components(n: int, edges: Sequence[Edge]) -> list[int]:
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for e in edges:
        ra, rb = find(e.i), find(e.j)
        if ra != rb:
            parent[ra] = rb
    roots = {}
    labels = []
    for i in range(n):
        r = find(i)
        labels.append(roots.setdefault(r, len(roots)))
    return labels


def cluster_entities(
    entities: Sequence[str],
    model,
    *,
    floor: float = 0.80,
    k: int = 5,
    require_mutual: bool = True,
    min_confidence: float | None = None,
    canonical: str = "shortest",
    force_cpu: bool = True,
) -> ClusterResult:
    """Cluster entity strings into canonical groups via mutual-NN merge edges.

    Args:
        k: Neighbourhood size; larger merges more aggressively.
        min_confidence: Drop edges below this before clustering.
        canonical: Cluster name: ``"shortest"`` member or ``"first"`` seen.
    """
    uniq = list(dict.fromkeys(str(e) for e in entities))
    embedder: Embedder = as_embedder(model, force_cpu=force_cpu)
    emb = np.asarray(embedder.encode(uniq))
    if emb.size and not np.isfinite(emb).all():
        raise ValueError(
            f"Embedder {getattr(embedder, 'name', 'unknown')!r} produced non-finite "
            "embeddings (NaN or inf); cannot cluster."
        )

    edges = mutual_nn_edges(emb, floor=floor, k=k, require_mutual=require_mutual)
    if min_confidence is not None:
        edges = [e for e in edges if e.confidence >= min_confidence]

    labels = _connected_components(len(uniq), edges)

    groups: dict[int, list[str]] = {}
    for ent, lab in zip(uniq, labels):
        groups.setdefault(lab, []).append(ent)
    canon: dict[int, str] = {}
    for lab, members in groups.items():
        if canonical == "first":
            canon[lab] = members[0]
        else:
            canon[lab] = min(members, key=lambda s: (len(s), s))

    return ClusterResult(
        entities=uniq,
        labels=labels,
        edges=edges,
        canonical=canon,
        embedder_name=getattr(embedder, "name", "unknown"),
    )
