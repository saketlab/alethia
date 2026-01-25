import logging
import os
import time
from typing import Any, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_VERBOSE_MODE = False


def set_verbose(verbose: bool = True, level: str = "INFO"):
    global _VERBOSE_MODE
    _VERBOSE_MODE = verbose
    log_level = (
        getattr(logging, level.upper(), logging.INFO) if verbose else logging.WARNING
    )
    logger.setLevel(log_level)
    logging.getLogger().setLevel(log_level)


# Check optional dependencies
def _check_dependencies():
    deps = {
        "SENTENCE_TRANSFORMERS_AVAILABLE": False,
        "FASTEMBED_AVAILABLE": False,
        "RAPIDFUZZ_AVAILABLE": False,
        "FAISS_AVAILABLE": False,
        "NUMBA_AVAILABLE": False,
        "OPENAI_AVAILABLE": False,
        "GEMINI_AVAILABLE": False,
    }

    try:
        import torch
        from sentence_transformers import SentenceTransformer

        deps["SENTENCE_TRANSFORMERS_AVAILABLE"] = True
        globals()["SentenceTransformer"] = SentenceTransformer
        globals()["torch"] = torch
    except ImportError:
        globals()["SentenceTransformer"] = None
        globals()["torch"] = None

    try:
        import fastembed

        deps["FASTEMBED_AVAILABLE"] = True
    except ImportError:
        pass

    try:
        from rapidfuzz import fuzz, process

        deps["RAPIDFUZZ_AVAILABLE"] = True
        globals()["fuzz"] = fuzz
        globals()["process"] = process
    except ImportError:
        pass

    try:
        import faiss

        deps["FAISS_AVAILABLE"] = True
    except ImportError:
        pass

    try:
        from numba import jit

        deps["NUMBA_AVAILABLE"] = True
        globals()["jit"] = jit
    except ImportError:

        def jit(nopython=True):
            return lambda func: func

        globals()["jit"] = jit

    try:
        from openai import OpenAI

        deps["OPENAI_AVAILABLE"] = True
        globals()["OpenAI"] = OpenAI
    except ImportError:
        globals()["OpenAI"] = None

    try:
        import google.generativeai as genai

        deps["GEMINI_AVAILABLE"] = True
        globals()["genai"] = genai
    except (ImportError, SyntaxError):
        globals()["genai"] = None

    return deps


DEPENDENCIES = _check_dependencies()
SENTENCE_TRANSFORMERS_AVAILABLE = DEPENDENCIES["SENTENCE_TRANSFORMERS_AVAILABLE"]
FASTEMBED_AVAILABLE = DEPENDENCIES["FASTEMBED_AVAILABLE"]
RAPIDFUZZ_AVAILABLE = DEPENDENCIES["RAPIDFUZZ_AVAILABLE"]
FAISS_AVAILABLE = DEPENDENCIES["FAISS_AVAILABLE"]
NUMBA_AVAILABLE = DEPENDENCIES["NUMBA_AVAILABLE"]
OPENAI_AVAILABLE = DEPENDENCIES["OPENAI_AVAILABLE"]
GEMINI_AVAILABLE = DEPENDENCIES["GEMINI_AVAILABLE"]


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


@jit(nopython=True)
def fast_cosine_similarity_matrix(A, B):
    return np.dot(A, B.T)


@jit(nopython=True)
def fast_normalize_embeddings(embeddings):
    norms = np.sqrt(np.sum(embeddings**2, axis=1))
    return embeddings / norms.reshape(-1, 1)


def get_fastembed_supported_models() -> list:
    if not FASTEMBED_AVAILABLE:
        return []
    from fastembed import TextEmbedding

    return [m["model"] for m in TextEmbedding.list_supported_models()]


def is_model_fastembed_supported(model_name: str) -> bool:
    if not FASTEMBED_AVAILABLE:
        return False
    return model_name in get_fastembed_supported_models()


def load_fastembed_model(model_name: str):
    if not FASTEMBED_AVAILABLE:
        raise ImportError(
            "FastEmbed not available. Install with: pip install fastembed"
        )
    if not is_model_fastembed_supported(model_name):
        return None

    from fastembed import TextEmbedding

    return TextEmbedding(model_name=model_name)


def load_sentence_transformer_model(
    model_name: str, force_cpu: bool = False, threads: int = 2
):
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "SentenceTransformers not available. Install with: pip install sentence-transformers"
        )

    device = "cpu" if force_cpu or not torch.cuda.is_available() else "cuda"

    if device == "cpu":
        if threads is None:
            import os

            threads = os.cpu_count() or 4
        torch.set_num_threads(threads)

    try:
        return SentenceTransformer(model_name, device=device, trust_remote_code=True)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            if threads is None:
                import os

                threads = os.cpu_count() or 4
            torch.set_num_threads(threads)
            return SentenceTransformer(model_name, device="cpu", trust_remote_code=True)
        raise


def get_best_available_backend(prefer_cpu: bool = False):
    if prefer_cpu and FASTEMBED_AVAILABLE:
        return "fastembed"
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        return "sentence-transformers"
    if FASTEMBED_AVAILABLE:
        return "fastembed"
    if OPENAI_AVAILABLE:
        return "openai"
    if GEMINI_AVAILABLE:
        return "gemini"
    if RAPIDFUZZ_AVAILABLE:
        return "rapidfuzz"
    return "exact"


def _get_openai_embedding(client, text: str, model: str = "text-embedding-ada-002"):
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding


def _get_gemini_embedding(text: str, model_name: str = "models/embedding-001"):
    result = genai.embed_content(model=model_name, content=text)
    return result["embedding"]


def run_openai_matching(
    dirty_entries: List[str],
    reference_entries: List[str],
    model_name: str = "text-embedding-ada-002",
    threshold: float = 0.7,
) -> pd.DataFrame:
    if not OPENAI_AVAILABLE:
        raise ImportError("OpenAI not available. Install with: pip install openai")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key)

    reference_embeddings = {
        ref: np.array(_get_openai_embedding(client, ref, model_name))
        for ref in reference_entries
    }

    results = []
    for entry in dirty_entries:
        if str(entry) == "nan":
            results.append(
                {
                    "given_entity": entry,
                    "alethia_prediction": np.nan,
                    "alethia_score": np.nan,
                }
            )
            continue

        query_emb = np.array(_get_openai_embedding(client, entry, model_name))
        similarities = {
            ref: cosine_similarity(query_emb, emb)
            for ref, emb in reference_embeddings.items()
        }
        best_match = max(similarities, key=similarities.get)
        best_score = similarities[best_match]

        if best_score >= threshold:
            results.append(
                {
                    "given_entity": entry,
                    "alethia_prediction": best_match,
                    "alethia_score": best_score,
                }
            )
        else:
            results.append(
                {
                    "given_entity": entry,
                    "alethia_prediction": pd.NA,
                    "alethia_score": pd.NA,
                }
            )

    return pd.DataFrame(results)


def run_gemini_matching(
    dirty_entries: List[str],
    reference_entries: List[str],
    model_name: str = "models/embedding-001",
    threshold: float = 0.7,
) -> pd.DataFrame:
    if not GEMINI_AVAILABLE:
        raise ImportError(
            "Google GenerativeAI not available. Install with: pip install google-generativeai"
        )

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")
    genai.configure(api_key=api_key)

    reference_embeddings = {
        ref: np.array(_get_gemini_embedding(ref, model_name))
        for ref in reference_entries
    }

    results = []
    for entry in dirty_entries:
        if str(entry) == "nan":
            results.append(
                {
                    "given_entity": entry,
                    "alethia_prediction": np.nan,
                    "alethia_score": np.nan,
                }
            )
            continue

        query_emb = np.array(_get_gemini_embedding(entry, model_name))
        similarities = {
            ref: cosine_similarity(query_emb, emb)
            for ref, emb in reference_embeddings.items()
        }
        best_match = max(similarities, key=similarities.get)
        best_score = similarities[best_match]

        if best_score >= threshold:
            results.append(
                {
                    "given_entity": entry,
                    "alethia_prediction": best_match,
                    "alethia_score": best_score,
                }
            )
        else:
            results.append(
                {
                    "given_entity": entry,
                    "alethia_prediction": entry,
                    "alethia_score": 1.0,
                }
            )

    return pd.DataFrame(results)


def run_rapidfuzz_matching(
    dirty_entries: List[str], reference_entries: List[str]
) -> pd.DataFrame:
    if not RAPIDFUZZ_AVAILABLE:
        raise ImportError(
            "RapidFuzz not available. Install with: pip install rapidfuzz"
        )

    results = []
    for entry in dirty_entries:
        match_result = process.extractOne(
            entry, reference_entries, scorer=fuzz.token_sort_ratio
        )
        if match_result:
            best_match, score, _ = match_result
            results.append(
                {
                    "given_entity": entry,
                    "alethia_prediction": best_match,
                    "alethia_score": score / 100,
                }
            )
        else:
            results.append(
                {
                    "given_entity": entry,
                    "alethia_prediction": entry,
                    "alethia_score": 1.0,
                }
            )

    return pd.DataFrame(results)


def _optimized_batch_matching(
    dirty_entries, reference_entries, model_obj, backend, threshold=0.7
):
    all_texts = dirty_entries + reference_entries

    if backend == "fastembed":
        all_embeddings = np.array(list(model_obj.embed(all_texts)))
    else:
        all_embeddings = model_obj.encode(
            all_texts, batch_size=64, show_progress_bar=_VERBOSE_MODE
        )

    n_dirty = len(dirty_entries)
    dirty_emb = all_embeddings[:n_dirty]
    ref_emb = all_embeddings[n_dirty:]

    if NUMBA_AVAILABLE:
        dirty_emb = fast_normalize_embeddings(dirty_emb)
        ref_emb = fast_normalize_embeddings(ref_emb)
        similarity_matrix = fast_cosine_similarity_matrix(dirty_emb, ref_emb)
    else:
        dirty_emb = dirty_emb / np.linalg.norm(dirty_emb, axis=1, keepdims=True)
        ref_emb = ref_emb / np.linalg.norm(ref_emb, axis=1, keepdims=True)
        similarity_matrix = np.dot(dirty_emb, ref_emb.T)

    best_indices = np.argmax(similarity_matrix, axis=1)
    best_scores = np.max(similarity_matrix, axis=1)

    results = []
    for entry, ref_idx, sim in zip(dirty_entries, best_indices, best_scores):
        if sim >= threshold:
            results.append(
                {
                    "given_entity": entry,
                    "alethia_prediction": reference_entries[ref_idx],
                    "alethia_score": float(sim),
                }
            )
        else:
            results.append(
                {
                    "given_entity": entry,
                    "alethia_prediction": entry,
                    "alethia_score": 1.0,
                }
            )

    return results


def _is_nan_entry(entry) -> bool:
    if entry is None or pd.isna(entry):
        return True
    if isinstance(entry, str) and entry.lower().strip() in [
        "nan",
        "null",
        "none",
        "",
        "na",
        "n/a",
    ]:
        return True
    return False


def _find_exact_matches(
    dirty_entries: List[str], reference_entries: List[str], case_sensitive: bool = False
):
    exact_matches = {}
    remaining = []
    remaining_indices = []

    if case_sensitive:
        ref_set = set(reference_entries)
    else:
        ref_lookup = {
            ref.lower(): ref for ref in reference_entries if isinstance(ref, str)
        }

    for i, entry in enumerate(dirty_entries):
        if _is_nan_entry(entry):
            remaining.append(entry)
            remaining_indices.append(i)
            continue

        found = False
        if case_sensitive:
            if entry in ref_set:
                exact_matches[i] = {
                    "given_entity": entry,
                    "alethia_prediction": entry,
                    "alethia_score": 1.0,
                }
                found = True
        else:
            if isinstance(entry, str) and entry.lower() in ref_lookup:
                exact_matches[i] = {
                    "given_entity": entry,
                    "alethia_prediction": ref_lookup[entry.lower()],
                    "alethia_score": 1.0,
                }
                found = True

        if not found:
            remaining.append(entry)
            remaining_indices.append(i)

    return exact_matches, remaining, remaining_indices


def alethia(
    dirty_entries: List[str],
    reference_entries: List[str],
    model: str = "auto",
    backend: str = "auto",
    force_cpu: bool = True,
    use_batch_optimization: bool = True,
    threshold: float = 0.7,
    verbose: bool = False,
    use_exact_matching: bool = True,
    exact_match_case_sensitive: bool = False,
    return_model_attrs: bool = True,
    drop_duplicates: bool = True,
    remove_identical_hits: bool = False,
    threads: int = 2,
    **kwargs,
) -> pd.DataFrame:
    """Main entity matching function."""
    old_verbose = _VERBOSE_MODE
    if verbose:
        set_verbose(True)

    try:
        start_time = time.time()

        # Preprocess: separate NaN entries
        processed, nan_mask, orig_indices = [], [], []
        for i, entry in enumerate(dirty_entries):
            is_nan = _is_nan_entry(entry)
            nan_mask.append(is_nan)
            if not is_nan:
                processed.append(entry)
                orig_indices.append(i)

        if not processed:
            return _create_nan_results(dirty_entries)

        clean_refs = [r for r in reference_entries if not _is_nan_entry(r)]
        if not clean_refs:
            return _create_no_ref_results(dirty_entries, nan_mask)

        # Exact matching phase
        exact_matches = {}
        remaining = processed
        remaining_indices = list(range(len(processed)))

        if use_exact_matching:
            exact_matches, remaining, remaining_indices = _find_exact_matches(
                processed, clean_refs, exact_match_case_sensitive
            )
            exact_matches = {orig_indices[k]: v for k, v in exact_matches.items()}

        if not remaining:
            return _build_final_results(
                exact_matches,
                pd.DataFrame(),
                [],
                dirty_entries,
                nan_mask,
                model,
                backend,
                start_time,
            )

        # Model-based matching
        if model == "rapidfuzz" or backend == "rapidfuzz":
            model_results = run_rapidfuzz_matching(remaining, clean_refs)
        elif model == "openai" or backend == "openai":
            model_results = run_openai_matching(
                remaining,
                clean_refs,
                kwargs.get("model_name", "text-embedding-ada-002"),
                threshold,
            )
        elif model == "gemini" or backend == "gemini":
            model_results = run_gemini_matching(
                remaining,
                clean_refs,
                kwargs.get("model_name", "models/embedding-001"),
                threshold,
            )
        else:
            if model == "auto":
                model = "sentence-transformers/all-MiniLM-L6-v2"

            has_gpu = (
                SENTENCE_TRANSFORMERS_AVAILABLE
                and torch is not None
                and torch.cuda.is_available()
                and not force_cpu
            )

            if backend == "auto":
                if has_gpu:
                    backend = "sentence-transformers"
                elif is_model_fastembed_supported(model):
                    backend = "fastembed"
                elif SENTENCE_TRANSFORMERS_AVAILABLE:
                    backend = "sentence-transformers"
                else:
                    backend = get_best_available_backend(prefer_cpu=force_cpu)

            if backend == "fastembed":
                model_obj = load_fastembed_model(model)
                if model_obj is None and SENTENCE_TRANSFORMERS_AVAILABLE:
                    backend = "sentence-transformers"
                    model_obj = load_sentence_transformer_model(
                        model, force_cpu=force_cpu, threads=threads
                    )
            else:
                model_obj = load_sentence_transformer_model(
                    model, force_cpu=force_cpu, threads=threads
                )

            if model_obj is None:
                raise ValueError(f"Failed to load model {model}")

            if use_batch_optimization and len(remaining) > 10:
                results = _optimized_batch_matching(
                    remaining, clean_refs, model_obj, backend, threshold
                )
            else:
                results = _standard_matching(
                    remaining, clean_refs, model_obj, backend, threshold
                )
            model_results = pd.DataFrame(results)

        df = _build_final_results(
            exact_matches,
            model_results,
            remaining_indices,
            dirty_entries,
            nan_mask,
            model,
            backend,
            start_time,
        )

        if return_model_attrs:
            df["alethia_method"] = model
            df["alethia_backend"] = backend
        if remove_identical_hits:
            df = df[df.given_entity != df.alethia_prediction]
        if drop_duplicates:
            df = df.drop_duplicates()

        return df

    finally:
        if not old_verbose:
            set_verbose(False)


def _standard_matching(
    dirty_entries, reference_entries, model_obj, backend, threshold=0.7
):
    if backend == "fastembed":
        ref_emb = {ref: next(model_obj.embed([ref])) for ref in reference_entries}
    else:
        ref_emb = {ref: model_obj.encode(ref) for ref in reference_entries}

    results = []
    for entry in dirty_entries:
        if str(entry) == "nan":
            results.append(
                {
                    "given_entity": entry,
                    "alethia_prediction": np.nan,
                    "alethia_score": np.nan,
                }
            )
            continue

        if backend == "fastembed":
            query_emb = next(model_obj.embed([entry]))
        else:
            query_emb = model_obj.encode(entry)

        similarities = {
            ref: cosine_similarity(query_emb, emb) for ref, emb in ref_emb.items()
        }
        best_match = max(similarities, key=similarities.get)
        best_score = similarities[best_match]

        if best_score >= threshold:
            results.append(
                {
                    "given_entity": entry,
                    "alethia_prediction": best_match,
                    "alethia_score": best_score,
                }
            )
        else:
            results.append(
                {
                    "given_entity": entry,
                    "alethia_prediction": entry,
                    "alethia_score": 1.0,
                }
            )

    return results


def _create_nan_results(entries):
    df = pd.DataFrame(
        [
            {"given_entity": e, "alethia_prediction": np.nan, "alethia_score": np.nan}
            for e in entries
        ]
    )
    df.attrs = {"backend": "none", "model": "none", "processing_time": 0.0}
    return df


def _create_no_ref_results(entries, nan_mask):
    results = []
    for i, entry in enumerate(entries):
        if nan_mask[i]:
            results.append(
                {
                    "given_entity": entry,
                    "alethia_prediction": np.nan,
                    "alethia_score": np.nan,
                }
            )
        else:
            results.append(
                {
                    "given_entity": entry,
                    "alethia_prediction": entry,
                    "alethia_score": 1.0,
                }
            )
    df = pd.DataFrame(results)
    df.attrs = {"backend": "none", "model": "none", "processing_time": 0.0}
    return df


def _build_final_results(
    exact_matches,
    model_results,
    remaining_indices,
    original_entries,
    nan_mask,
    model,
    backend,
    start_time,
):
    results = []
    model_idx = 0

    for i, entry in enumerate(original_entries):
        if nan_mask[i]:
            results.append(
                {
                    "given_entity": entry,
                    "alethia_prediction": np.nan,
                    "alethia_score": np.nan,
                }
            )
        elif i in exact_matches:
            results.append(exact_matches[i])
        elif model_idx < len(model_results):
            row = model_results.iloc[model_idx].to_dict()
            row["given_entity"] = entry
            results.append(row)
            model_idx += 1
        else:
            results.append(
                {
                    "given_entity": entry,
                    "alethia_prediction": entry,
                    "alethia_score": 1.0,
                }
            )

    df = pd.DataFrame(results)
    df.attrs = {
        "backend": backend,
        "model": model,
        "processing_time": time.time() - start_time,
        "exact_matches_count": len(exact_matches),
        "nan_entries_count": sum(nan_mask),
    }
    return df
