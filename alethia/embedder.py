"""Model-agnostic embedding seam: normalize a name or callable into an Embedder."""

import logging
from typing import Any, Callable, List, Optional, Sequence, Union

import numpy as np

from .cpu_optimizations import cpu_thread_pool, get_cpu_runtime_hints

logger = logging.getLogger(__name__)

# Keywords routed to the string-similarity / API paths, not treated as embedding models
# in ``alethia()`` and therefore must not be resolved to an Embedder.
NON_EMBEDDING_MODELS = {"rapidfuzz", "exact", "fuzzy", "instructor"}

EmbedFn = Callable[[List[str]], Any]
# A model can be given as a name string, an embed_fn callable, an Embedder, or a loaded
# model object. ``Any`` covers the latter two; kept explicit for documentation.
ModelSpec = Union[str, EmbedFn, Any]


class Embedder:
    """Abstract embedding source.

    Subclasses implement :meth:`encode`. Instances expose ``name`` (model identity, for
    reporting), ``family`` (the backend family: ``callable``, ``sentence-transformers``,
    ``fastembed``), and, after the first encode, ``dim`` (the embedding dimensionality).
    """

    name: str = "embedder"
    family: str = "embedder"
    dim: Optional[int] = None

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
    """Wrap a user-supplied ``embed_fn(list[str]) -> array-like``.

    The function may return a 2D array, a list of vectors, or (for a single text) a 1D
    vector; output is coerced to ``np.ndarray`` of shape ``(n, d)``.
    """

    family = "callable"

    def __init__(self, embed_fn: EmbedFn, name: Optional[str] = None) -> None:
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
            # A flat vector for many inputs is ambiguous; reject loudly.
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


def l2_normalize(arr: np.ndarray, dtype=np.float32) -> np.ndarray:
    """Row-wise L2 normalization, safe for zero vectors.

    This is the single canonical normalization used across matching and assessment.
    ``dtype`` defaults to float32 (matching path); the assessor passes float64 for
    metric precision.
    """
    arr = np.asarray(arr, dtype=dtype)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def match_by_embeddings(
    queries: Sequence[str],
    references: Sequence[str],
    embedder: Embedder,
    threshold: Optional[float] = None,
) -> List[dict]:
    """Match each query to its most similar reference using cosine similarity.

    Embeds queries and references with ``embedder``, L2-normalizes, and returns, per
    query, the best reference and its cosine score. This is the single embedding-based
    matching primitive shared by every embedding backend.

    Args:
        queries: Query strings.
        references: Reference strings.
        embedder: The :class:`Embedder` to encode with.
        threshold: If given, matches scoring below it yield a ``NaN`` prediction/score
            (the query had no sufficiently-similar reference). If ``None`` (default), the
            best match is always returned regardless of score.

    Returns a list of ``{given_entity, alethia_prediction, alethia_score}`` dicts.
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
    best_idx = np.argmax(sims, axis=1)
    best_score = np.max(sims, axis=1)

    results = []
    for q, idx, score in zip(queries, best_idx, best_score):
        if threshold is not None and score < threshold:
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
    name: Optional[str] = None,
) -> Embedder:
    """Normalize ``model`` into an :class:`Embedder`.

    Args:
        model: A callable ``embed_fn``, an already-constructed :class:`Embedder`, a loaded
            model object (SentenceTransformer / FastEmbed), or a model **name string**.
        backend: Backend hint for name resolution (``"auto"``, ``"sentence-transformers"``,
            ``"fastembed"``). Ignored for callables / objects.
        force_cpu: Prefer CPU when loading a named model.
        name: Optional display name override.

    Returns:
        An :class:`Embedder`.

    Raises:
        ValueError: if ``model`` is a non-embedding keyword (e.g. ``"rapidfuzz"``), which is
            handled outside the embedding path.
        TypeError / ImportError: on unresolvable specs or missing backends.
    """
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

    # A pre-loaded model object, dispatched by its interface. Checked before the bare
    # callable case because SentenceTransformer is itself callable but must use .encode.
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
    name: Optional[str],
) -> Embedder:
    """Resolve a model name to an Embedder via the backend registry.

    Imports are done lazily here so that naming a model never forces a heavy dependency
    at module import time.
    """
    # Local import avoids a circular import with alethia.alethia at module load.
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
        return FastEmbedEmbedder(obj, name=name or model_name) if obj is not None else None

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
        # FastEmbed is fast but only catalogs a subset of models; sentence-transformers
        # resolves any HuggingFace sentence model. Try FastEmbed first, then fall back.
        emb = try_fastembed() or try_sentence_transformers()
        if emb is None:
            raise ImportError(
                f"Could not load model '{model_name}'. Install sentence-transformers or "
                "fastembed, or pass an embed_fn callable."
            )
        return emb

    raise ValueError(f"Unknown or non-embedding backend: {backend!r}")
