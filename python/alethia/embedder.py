"""Model-agnostic embedding seam: normalize a name or callable into an Embedder."""

import logging
from collections.abc import Callable, Sequence
from typing import Any, Union

import numpy as np

from .cpu_optimizations import cpu_thread_pool, get_cpu_runtime_hints

logger = logging.getLogger(__name__)

# routed to the string-similarity and API paths, never to an Embedder
NON_EMBEDDING_MODELS = {
    "rapidfuzz",
    "exact",
    "fuzzy",
    "instructor",
    "openai",
    "gemini",
}

EmbedFn = Callable[[list[str]], Any]
ModelSpec = Union[str, EmbedFn, Any]  # noqa: UP007 - runtime alias, not an annotation


class Embedder:
    """Abstract embedding source exposing its name, family, and dimension."""

    name: str = "embedder"
    family: str = "embedder"
    dim: int | None = None

    def encode(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def _finalize(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        self.dim = int(arr.shape[1])
        return arr

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, dim={self.dim})"


class CallableEmbedder(Embedder):
    """Wrap an embedding callable and coerce its output to an ``(n, d)`` array."""

    family = "callable"

    def __init__(self, embed_fn: EmbedFn, name: str | None = None) -> None:
        if not callable(embed_fn):
            raise TypeError("embed_fn must be callable")
        self._fn = embed_fn
        self.name = name or getattr(embed_fn, "__name__", "callable")

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.empty((0, self.dim or 0), dtype=np.float32)
        out = self._fn(texts)
        arr = np.asarray(out, dtype=np.float32)
        if arr.ndim == 1 and len(texts) > 1:
            raise ValueError(
                f"embed_fn returned a 1D array for {len(texts)} inputs; "
                "expected shape (n, d)."
            )
        return self._finalize(arr)


class SentenceTransformerEmbedder(Embedder):
    """Embedder backed by a loaded SentenceTransformer model object."""

    family = "sentence-transformers"

    def __init__(self, model_obj: Any, name: str = "sentence-transformers") -> None:
        self._model = model_obj
        self.name = name

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.empty((0, self.dim or 0), dtype=np.float32)
        hints = get_cpu_runtime_hints(self._model)
        batch_size = hints.get("batch_size") or 64
        with cpu_thread_pool(hints.get("num_threads")):
            arr = self._model.encode(
                texts, batch_size=batch_size, show_progress_bar=False
            )
        return self._finalize(arr)


class FastEmbedEmbedder(Embedder):
    """Embedder backed by a FastEmbed ``TextEmbedding`` model object."""

    family = "fastembed"

    def __init__(self, model_obj: Any, name: str = "fastembed") -> None:
        self._model = model_obj
        self.name = name

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.empty((0, self.dim or 0), dtype=np.float32)
        arr = np.array(list(self._model.embed(texts)))
        return self._finalize(arr)


class CachingEmbedder(Embedder):
    """Memoize another embedder's output by exact string for the life of a run."""

    def __init__(self, inner: "Embedder", name: str | None = None) -> None:
        if isinstance(inner, CachingEmbedder):
            self._inner = inner._inner
            self._cache = inner._cache
            self.name = name or inner.name
            self.family = inner.family
            return
        self._inner = inner
        self.name = name or inner.name
        self.family = inner.family
        self._cache: dict = {}

    @property
    def dim(self) -> int | None:  # type: ignore[override]
        return self._inner.dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.empty((0, self._inner.dim or 0), dtype=np.float32)

        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            fresh = self._inner.encode(missing)
            for text, vec in zip(missing, fresh):
                # vec is a view holding the whole batch array alive
                self._cache[text] = np.ascontiguousarray(vec)
        return np.stack([self._cache[t] for t in texts]).astype(np.float32, copy=False)

    def clear(self) -> None:
        """Drop cached vectors, for a wrapper spanning several reference sets."""
        self._cache.clear()


# unrounded, ties turn on dot-product last bits, which differ between BLAS builds
RANK_DECIMALS = 12


def rank_key(sims: np.ndarray) -> np.ndarray:
    """Similarities quantized for a reproducible ordering. See RANK_DECIMALS."""
    return np.round(sims, RANK_DECIMALS)


def top_k_stable(sims: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k best per row, descending. Ties break by ascending index.

    argpartition would resolve a tie at the kth position arbitrarily.
    """
    return np.argsort(-rank_key(sims), axis=1, kind="stable")[:, :k]


def l2_normalize(arr: np.ndarray, dtype=np.float64) -> np.ndarray:
    """Canonical row-wise L2 normalization; float64 keeps tie-breaks build-independent."""
    arr = np.asarray(arr, dtype=dtype)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def match_by_embeddings(
    queries: Sequence[str],
    references: Sequence[str],
    embedder: Embedder,
    threshold: float | None = None,
) -> list[dict]:
    """Match queries to references by cosine similarity.

    The single embedding-based matching primitive; below ``threshold`` a query yields a
    ``NaN`` prediction and score rather than its best available match.
    """
    queries = list(queries)
    references = list(references)
    if not queries or not references:
        return [
            {"given_entity": q, "alethia_prediction": np.nan, "alethia_score": np.nan}
            for q in queries
        ]

    q_emb = l2_normalize(embedder.encode(queries))
    r_emb = l2_normalize(embedder.encode(references))
    sims = q_emb @ r_emb.T  # (n_queries, n_refs)
    # rank on rounded values, report the unrounded score
    best_idx = np.argmax(rank_key(sims), axis=1)
    best_score = sims[np.arange(sims.shape[0]), best_idx]

    results = []
    accept = None if threshold is None else rank_key(best_score) >= rank_key(threshold)
    for n, (q, idx, score) in enumerate(zip(queries, best_idx, best_score)):
        if accept is not None and not accept[n]:
            results.append(
                {
                    "given_entity": q,
                    "alethia_prediction": np.nan,
                    "alethia_score": np.nan,
                }
            )
        else:
            results.append(
                {
                    "given_entity": q,
                    "alethia_prediction": references[int(idx)],
                    "alethia_score": float(score),
                }
            )
    return results


def as_embedder(
    model: ModelSpec,
    *,
    backend: str = "auto",
    force_cpu: bool = True,
    name: str | None = None,
) -> Embedder:
    """Normalize a callable, loaded model, model name, or embedder into an Embedder."""
    if isinstance(model, Embedder):
        return model

    if isinstance(model, str):
        key = model.strip().lower()
        if key in NON_EMBEDDING_MODELS:
            raise ValueError(
                f"'{model}' is not an embedding model; it is handled by the "
                "string-similarity / API path, not as an Embedder."
            )
        return _resolve_named_embedder(
            model, backend=backend, force_cpu=force_cpu, name=name
        )

    # SentenceTransformer is callable, so this must precede the callable case
    if hasattr(model, "embed") and not hasattr(model, "encode"):  # FastEmbed
        return FastEmbedEmbedder(model, name=name or "fastembed")
    if hasattr(model, "encode"):  # SentenceTransformer and HF sentence models
        return SentenceTransformerEmbedder(model, name=name or "sentence-transformers")

    if callable(model):  # a plain embed_fn(list[str]) -> ndarray
        return CallableEmbedder(model, name=name)

    raise TypeError(
        f"Cannot interpret model of type {type(model).__name__} as an embedder. "
        "Pass a callable embed_fn, a model name string, or a loaded model object."
    )


def _resolve_named_embedder(
    model_name: str,
    *,
    backend: str,
    force_cpu: bool,
    name: str | None,
) -> Embedder:
    """Resolve a model name to an Embedder, importing backends lazily."""
    # circular: alethia.alethia imports this module
    from .alethia import (
        FASTEMBED_AVAILABLE,
        SENTENCE_TRANSFORMERS_AVAILABLE,
        load_fastembed_model,
        load_sentence_transformer_model,
    )

    def try_fastembed():
        if not FASTEMBED_AVAILABLE:
            return None
        try:
            obj = load_fastembed_model(model_name)
        except Exception:
            return None
        return (
            FastEmbedEmbedder(obj, name=name or model_name) if obj is not None else None
        )

    def try_sentence_transformers():
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            return None
        obj = load_sentence_transformer_model(model_name, force_cpu=force_cpu)
        return (
            SentenceTransformerEmbedder(obj, name=name or model_name)
            if obj is not None
            else None
        )

    if backend == "fastembed":
        emb = try_fastembed()
        if emb is None:
            raise ValueError(f"Failed to load '{model_name}' with FastEmbed.")
        return emb

    if backend == "sentence-transformers":
        emb = try_sentence_transformers()
        if emb is None:
            raise ValueError(
                f"Failed to load '{model_name}' with sentence-transformers."
            )
        return emb

    if backend == "auto":
        # fastembed first: smaller download and faster CPU start, fewer models
        emb = try_fastembed() or try_sentence_transformers()
        if emb is None:
            raise ImportError(
                f"Could not load model '{model_name}'. Install sentence-transformers or "
                "fastembed, or pass an embed_fn callable."
            )
        return emb

    raise ValueError(f"Unknown or non-embedding backend: {backend!r}")
