import importlib.util
import logging
import os
import time
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from .utils import print_resource_usage

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_VERBOSE_MODE = False


def set_verbose(verbose: bool = True, level: str = "INFO"):
    """Turn package logging on or off at ``level``."""
    global _VERBOSE_MODE
    _VERBOSE_MODE = verbose

    if verbose:
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
        }
        logger.setLevel(level_map.get(level.upper(), logging.INFO))
        logging.getLogger().setLevel(level_map.get(level.upper(), logging.INFO))
    else:
        logger.setLevel(logging.WARNING)
        logging.getLogger().setLevel(logging.WARNING)


def _is_installed(module_name: str) -> bool:
    """Whether ``module_name`` is importable, judged without importing it."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        # a parent package that fails to import raises rather than returning None
        return False


_DEPENDENCY_MODULES = {
    "SENTENCE_TRANSFORMERS_AVAILABLE": "sentence_transformers",
    "FASTEMBED_AVAILABLE": "fastembed",
    "RAPIDFUZZ_AVAILABLE": "rapidfuzz",
    "OPENAI_AVAILABLE": "openai",
    "GEMINI_AVAILABLE": "google.generativeai",
}


def check_optional_dependencies(verbose: bool = False):
    """Map each availability flag to whether its backend is importable."""
    original_level = logger.level
    if verbose or _VERBOSE_MODE:
        logger.setLevel(logging.INFO)

    dependencies = {
        flag: _is_installed(module) for flag, module in _DEPENDENCY_MODULES.items()
    }

    if verbose or _VERBOSE_MODE:
        for flag, available in dependencies.items():
            name = flag.replace("_AVAILABLE", "").replace("_", " ").title()
            logger.info("%s %s", "[ok]" if available else "[x]", name)

    logger.setLevel(original_level)

    unavailable_core = []
    if (
        not dependencies["SENTENCE_TRANSFORMERS_AVAILABLE"]
        and not dependencies["FASTEMBED_AVAILABLE"]
    ):
        unavailable_core.append("embedding models (sentence-transformers or fastembed)")
    if not dependencies["RAPIDFUZZ_AVAILABLE"]:
        unavailable_core.append("fuzzy matching (rapidfuzz)")

    if unavailable_core:
        logger.info(
            "Missing optional dependencies: %s. Install with: "
            "pip install 'alethia[recommended]'",
            ", ".join(unavailable_core),
        )

    return dependencies


# embedding and API backends only; rapidfuzz ratios are a different scale
DEFAULT_THRESHOLD = 0.7

DEPENDENCIES = check_optional_dependencies()
SENTENCE_TRANSFORMERS_AVAILABLE = DEPENDENCIES["SENTENCE_TRANSFORMERS_AVAILABLE"]
FASTEMBED_AVAILABLE = DEPENDENCIES["FASTEMBED_AVAILABLE"]
RAPIDFUZZ_AVAILABLE = DEPENDENCIES["RAPIDFUZZ_AVAILABLE"]
OPENAI_AVAILABLE = DEPENDENCIES["OPENAI_AVAILABLE"]
GEMINI_AVAILABLE = DEPENDENCIES["GEMINI_AVAILABLE"]

try:
    from .utils import print_resource_usage
except ImportError:

    def print_resource_usage():
        """Fallback function if utils not available"""
        try:
            import psutil

            print(f"Memory usage: {psutil.virtual_memory().percent}%")
        except ImportError:
            if _VERBOSE_MODE:
                logger.debug("psutil not available for resource monitoring")


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def get_openai_embedding(client, text: str, model: str = "text-embedding-ada-002"):
    """Get embedding from OpenAI API"""
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding


def get_gemini_embedding(text: str, model_name: str = "models/embedding-001"):
    """Get embedding from Gemini API"""
    import google.generativeai as genai

    result = genai.embed_content(model=model_name, content=text)
    return result["embedding"]


def load_fastembed_model(model_name: str):
    """Load FastEmbed model with proper error handling"""
    if not FASTEMBED_AVAILABLE:
        raise ImportError(
            "FastEmbed not available. Install with: pip install fastembed"
        )

    try:
        from fastembed import TextEmbedding

        model_mapping = {
            "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
            "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
        }

        fastembed_model_name = model_mapping.get(model_name, model_name)

        available_models = [m["model"] for m in TextEmbedding.list_supported_models()]

        if fastembed_model_name not in available_models:
            logger.error(f"Model '{fastembed_model_name}' not available in FastEmbed")
            if _VERBOSE_MODE:
                logger.info("Available models:")
                for model in available_models[:10]:
                    logger.info(f"  - {model}")
                if len(available_models) > 10:
                    logger.info(f"  ... and {len(available_models) - 10} more")
            return None

        model = TextEmbedding(model_name=fastembed_model_name)
        if _VERBOSE_MODE:
            logger.info(
                f"[ok] Successfully loaded {fastembed_model_name} with FastEmbed"
            )
        return model

    except Exception as e:
        logger.error(f"Error loading FastEmbed model: {e}")
        return None


def load_sentence_transformer_model(model_name: str, force_cpu: bool = False):
    """Load SentenceTransformer model with proper error handling"""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "SentenceTransformers not available. Install with: pip install sentence-transformers"
        )

    import torch
    from sentence_transformers import SentenceTransformer

    try:
        if force_cpu or not torch.cuda.is_available():
            device = "cpu"
            if _VERBOSE_MODE:
                logger.info(f"Loading {model_name} on CPU")
        else:
            device = "cuda"
            if _VERBOSE_MODE:
                logger.info(f"Loading {model_name} on GPU")

        model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
        if _VERBOSE_MODE:
            logger.info(f"[ok] Successfully loaded {model_name}")
        return model

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            logger.warning("GPU memory error, trying CPU")
            try:
                model = SentenceTransformer(
                    model_name, device="cpu", trust_remote_code=True
                )
                if _VERBOSE_MODE:
                    logger.info(f"[ok] Successfully loaded {model_name} on CPU")
                return model
            except Exception as cpu_e:
                logger.error(f"Failed to load on CPU: {cpu_e}")
                return None
        else:
            logger.error(f"Error loading model: {e}")
            return None
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None


def setup_openai_client():
    """Setup OpenAI client with API key from environment"""
    if not OPENAI_AVAILABLE:
        raise ImportError("OpenAI not available. Install with: pip install openai")

    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        raise ValueError("OPENAI_API_KEY not set in environment variables.")

    return OpenAI(api_key=api_key)


def setup_gemini_client():
    """Setup Gemini client with API key from environment"""
    if not GEMINI_AVAILABLE:
        raise ImportError(
            "Google GenerativeAI not available. Install with: pip install google-generativeai"
        )

    import google.generativeai as genai

    api_key = os.getenv("GEMINI_API_KEY")
    if api_key is None:
        raise ValueError("GEMINI_API_KEY not set in environment variables.")

    genai.configure(api_key=api_key)
    return genai


def get_best_available_backend(prefer_cpu: bool = False):
    """Name of the best installed embedding backend."""
    if prefer_cpu and FASTEMBED_AVAILABLE:
        return "fastembed"
    elif SENTENCE_TRANSFORMERS_AVAILABLE:
        return "sentence-transformers"
    elif FASTEMBED_AVAILABLE:
        return "fastembed"
    elif OPENAI_AVAILABLE:
        return "openai"
    elif GEMINI_AVAILABLE:
        return "gemini"
    elif RAPIDFUZZ_AVAILABLE:
        return "rapidfuzz"
    else:
        return "exact"


def _no_match_row(entity: Any) -> dict[str, Any]:
    """The row shape every backend emits when no reference is acceptable."""
    return {
        "given_entity": entity,
        "alethia_prediction": np.nan,
        "alethia_score": np.nan,
    }


def _run_api_matching(
    dirty_entries: list[str],
    reference_entries: list[str],
    embed: Any,
    label: str,
    threshold: float,
) -> pd.DataFrame:
    """Match by embedding one string at a time through ``embed``."""
    if _VERBOSE_MODE:
        logger.info(f"Computing {label} reference embeddings...")

    def _iter(items, desc):
        return tqdm(items, desc=desc) if (_VERBOSE_MODE or len(items) > 20) else items

    reference_embeddings = {
        ref: np.array(embed(ref))
        for ref in _iter(reference_entries, "Reference embeddings")
    }

    if _VERBOSE_MODE:
        logger.info(f"Processing queries with {label}...")

    results = []
    for incorrect in _iter(dirty_entries, "Processing queries"):
        if str(incorrect) == "nan":
            results.append(_no_match_row(incorrect))
            continue

        query_embedding = np.array(embed(incorrect))
        similarities = {
            ref: cosine_similarity(query_embedding, ref_embedding)
            for ref, ref_embedding in reference_embeddings.items()
        }

        best_match = max(similarities, key=similarities.get)
        best_score = similarities[best_match]

        if best_score >= threshold:
            results.append(
                {
                    "given_entity": incorrect,
                    "alethia_prediction": best_match,
                    "alethia_score": best_score,
                }
            )
        else:
            results.append(_no_match_row(incorrect))

    return pd.DataFrame(results)


def run_openai_matching(
    dirty_entries: list[str],
    reference_entries: list[str],
    model_name: str = "text-embedding-ada-002",
    threshold: float = 0.7,
) -> pd.DataFrame:
    """Run OpenAI-based matching"""
    if not OPENAI_AVAILABLE:
        raise ImportError("OpenAI not available. Install with: pip install openai")

    client = setup_openai_client()
    return _run_api_matching(
        dirty_entries,
        reference_entries,
        lambda text: get_openai_embedding(client, text, model_name),
        "OpenAI",
        threshold,
    )


def run_gemini_matching(
    dirty_entries: list[str],
    reference_entries: list[str],
    model_name: str = "models/embedding-001",
    threshold: float = 0.7,
) -> pd.DataFrame:
    """Run Gemini-based matching"""
    if not GEMINI_AVAILABLE:
        raise ImportError(
            "Google GenerativeAI not available. Install with: pip install google-generativeai"
        )

    setup_gemini_client()
    return _run_api_matching(
        dirty_entries,
        reference_entries,
        lambda text: get_gemini_embedding(text, model_name),
        "Gemini",
        threshold,
    )


def optimized_batch_matching(
    dirty_entries, reference_entries, model_obj, backend, threshold=DEFAULT_THRESHOLD
):
    """Match through the embedding seam, so the tie-break matches every other path."""
    from .embedder import as_embedder, match_by_embeddings

    return match_by_embeddings(
        dirty_entries, reference_entries, as_embedder(model_obj), threshold
    )


def standard_matching(
    dirty_entries, reference_entries, model_obj, backend, threshold=DEFAULT_THRESHOLD
):
    """Kept for callers that named it; identical to optimized_batch_matching."""
    return optimized_batch_matching(
        dirty_entries, reference_entries, model_obj, backend, threshold
    )


def _run_fallback(
    dirty_entries: list[str],
    reference_entries: list[str],
    failed_backend: str,
    threshold: float | None,
    scored_threshold: float,
):
    """Match with the best remaining backend after ``failed_backend`` gave out."""
    candidates = (
        (
            "rapidfuzz",
            RAPIDFUZZ_AVAILABLE,
            lambda: run_rapidfuzz_matching(dirty_entries, reference_entries, threshold),
        ),
        (
            "openai",
            OPENAI_AVAILABLE,
            lambda: run_openai_matching(
                dirty_entries,
                reference_entries,
                "text-embedding-ada-002",
                scored_threshold,
            ),
        ),
        (
            "gemini",
            GEMINI_AVAILABLE,
            lambda: run_gemini_matching(
                dirty_entries,
                reference_entries,
                "models/embedding-001",
                scored_threshold,
            ),
        ),
    )
    for name, available, run in candidates:
        if name != failed_backend and available:
            logger.info(f"Falling back to {name}")
            return run(), name
    raise


def apply_threshold(results: pd.DataFrame, threshold: float | None) -> pd.DataFrame:
    """Blank out predictions scoring below ``threshold``."""
    if threshold is None or results.empty:
        return results
    below = results["alethia_score"] < threshold
    results.loc[below, ["alethia_prediction", "alethia_score"]] = np.nan
    return results


def run_rapidfuzz_matching(
    dirty_entries: list[str],
    reference_entries: list[str],
    threshold: float | None = None,
) -> pd.DataFrame:
    """Match each entry to its best reference by RapidFuzz string similarity."""
    if not RAPIDFUZZ_AVAILABLE:
        raise ImportError(
            "RapidFuzz not available. Install with: pip install rapidfuzz"
        )

    from rapidfuzz import fuzz, process

    results = []
    iterator = (
        tqdm(dirty_entries, desc="RapidFuzz matching")
        if (_VERBOSE_MODE or len(dirty_entries) > 20)
        else dirty_entries
    )
    for incorrect in iterator:
        match_result = process.extractOne(
            incorrect, reference_entries, scorer=fuzz.token_sort_ratio
        )
        if match_result is not None:
            best_match, score, _ = match_result
            results.append(
                {
                    "given_entity": incorrect,
                    "alethia_prediction": best_match,
                    "alethia_score": score / 100,
                }
            )
        else:
            # extractOne returns None when there is nothing to match against
            results.append(_no_match_row(incorrect))

    return apply_threshold(pd.DataFrame(results), threshold)


def _find_exact_matches(
    dirty_entries: list[str],
    reference_entries: list[str],
    case_sensitive: bool = False,
    verbose: bool = False,
) -> tuple:
    """Split off entries already matching a reference; returns (matches, rest, indices)."""
    exact_matches = {}
    remaining_dirty_entries = []
    remaining_indices = []

    if case_sensitive:
        reference_set = set(reference_entries)
    else:
        reference_lookup = {
            ref.lower(): ref for ref in reference_entries if isinstance(ref, str)
        }

    for i, dirty_entry in enumerate(dirty_entries):
        if _is_nan_entry(dirty_entry):
            remaining_dirty_entries.append(dirty_entry)
            remaining_indices.append(i)
            continue

        found_exact_match = False

        if case_sensitive:
            if dirty_entry in reference_set:
                exact_matches[i] = {
                    "given_entity": dirty_entry,
                    "alethia_prediction": dirty_entry,
                    "alethia_score": 1.0,
                }
                found_exact_match = True
        else:
            if isinstance(dirty_entry, str):
                dirty_lower = dirty_entry.lower()
                if dirty_lower in reference_lookup:
                    exact_matches[i] = {
                        "given_entity": dirty_entry,
                        "alethia_prediction": reference_lookup[dirty_lower],
                        "alethia_score": 1.0,
                    }
                    found_exact_match = True

        if not found_exact_match:
            remaining_dirty_entries.append(dirty_entry)
            remaining_indices.append(i)

    if verbose:
        exact_count = len(exact_matches)
        remaining_count = len(remaining_dirty_entries)
        total_count = len(dirty_entries)
        logger.info(
            f"Exact matches: {exact_count} found, {remaining_count} remaining out of {total_count} total"
        )

    return exact_matches, remaining_dirty_entries, remaining_indices


def alethia(
    dirty_entries: list[str],
    reference_entries: list[str],
    model: str = "rapidfuzz",
    backend: str = "auto",
    force_cpu: bool = True,
    use_batch_optimization: bool = True,
    threshold: float | None = None,
    verbose: bool = False,
    use_exact_matching: bool = True,
    exact_match_case_sensitive: bool = False,
    return_model_attrs: bool = True,
    drop_duplicates: bool = True,
    remove_identical_hits: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Match each dirty entry to its closest reference entry."""
    old_verbose = _VERBOSE_MODE
    if verbose:
        set_verbose(True)

    try:
        if verbose or _VERBOSE_MODE:
            logger.info(f"Running Alethia with model: {model}, backend: {backend}")
            if use_exact_matching:
                logger.info(
                    f"Exact matching enabled (case_sensitive={exact_match_case_sensitive})"
                )

        start_time = time.time()
        if verbose or _VERBOSE_MODE:
            print("Initial resource usage:")
            print_resource_usage()

        processed_entries, nan_mask, original_indices = _preprocess_entries_with_nans(
            dirty_entries, verbose or _VERBOSE_MODE
        )

        if len(processed_entries) == 0:
            if verbose or _VERBOSE_MODE:
                logger.info("All entries are NaN, returning NaN results")
            return _create_nan_only_results(dirty_entries)

        clean_reference_entries = _filter_nan_entries(
            reference_entries, verbose or _VERBOSE_MODE
        )

        if len(clean_reference_entries) == 0:
            if verbose or _VERBOSE_MODE:
                logger.warning("All reference entries are NaN, cannot perform matching")
            return _create_no_match_results(dirty_entries)

        exact_matches = {}
        remaining_for_model = processed_entries
        remaining_indices_for_model = list(range(len(processed_entries)))

        if use_exact_matching:
            exact_matches, remaining_for_model, remaining_indices_for_model = (
                _find_exact_matches(
                    processed_entries,
                    clean_reference_entries,
                    case_sensitive=exact_match_case_sensitive,
                    verbose=verbose or _VERBOSE_MODE,
                )
            )

            original_exact_matches = {}
            for proc_idx, match_result in exact_matches.items():
                original_idx = original_indices[proc_idx]
                original_exact_matches[original_idx] = match_result
            exact_matches = original_exact_matches

        if len(remaining_for_model) == 0:
            if verbose or _VERBOSE_MODE:
                logger.info(
                    "All valid entries were exact matches, no model processing needed"
                )

            final_results = _reconstruct_results_with_exact_matches(
                exact_matches, dirty_entries, nan_mask, verbose or _VERBOSE_MODE
            )

            processing_time = time.time() - start_time
            final_results.attrs.update(
                {
                    "acceleration": "Exact-only",
                    "backend": "exact",
                    "effective_backend": "exact",
                    "processing_time": processing_time,
                    "model": "exact",
                    "nan_entries_count": sum(nan_mask),
                    "exact_matches_count": len(exact_matches),
                    "processed_entries_count": 0,
                }
            )

            if return_model_attrs:
                final_results["alethia_method"] = "exact"
                final_results["alethia_backend"] = "exact"
            return final_results

        if verbose or _VERBOSE_MODE:
            logger.info(
                f"Processing {len(remaining_for_model)} entries through model matching"
            )

        # rapidfuzz keeps the raw threshold, where None means best guess, unscored
        scored_threshold = DEFAULT_THRESHOLD if threshold is None else threshold

        # backend holds the selection, effective_backend what ran
        effective_backend = backend
        fallback_from: str | None = None

        if model == "rapidfuzz" or backend == "rapidfuzz":
            model_results = run_rapidfuzz_matching(
                remaining_for_model, clean_reference_entries, threshold
            )
            backend = effective_backend = "rapidfuzz"
        elif model == "openai" or backend == "openai":
            model_name = kwargs.get("model_name", "text-embedding-ada-002")
            model_results = run_openai_matching(
                remaining_for_model,
                clean_reference_entries,
                model_name,
                scored_threshold,
            )
            backend = effective_backend = "openai"
        elif model == "gemini" or backend == "gemini":
            model_name = kwargs.get("model_name", "models/embedding-001")
            model_results = run_gemini_matching(
                remaining_for_model,
                clean_reference_entries,
                model_name,
                scored_threshold,
            )
            backend = effective_backend = "gemini"
        elif not isinstance(model, str):
            from .embedder import as_embedder, match_by_embeddings

            embedder_obj = as_embedder(model, force_cpu=force_cpu)
            backend = effective_backend = embedder_obj.family
            if verbose or _VERBOSE_MODE:
                logger.info(f"Resolved non-string model spec to backend: {backend}")
            model_results = pd.DataFrame(
                match_by_embeddings(
                    remaining_for_model,
                    clean_reference_entries,
                    embedder_obj,
                    scored_threshold,
                )
            )
            acceleration = "Embedder"
        else:
            if backend == "auto":
                backend = get_best_available_backend(prefer_cpu=force_cpu)
                if verbose or _VERBOSE_MODE:
                    logger.info(f"Auto-selected backend: {backend}")
            effective_backend = backend

            try:
                if backend == "fastembed":
                    model_obj = load_fastembed_model(model)
                elif backend == "sentence-transformers":
                    model_obj = load_sentence_transformer_model(
                        model, force_cpu=force_cpu
                    )
                else:
                    raise ValueError(f"Unsupported backend: {backend}")

                if model_obj is None:
                    raise ValueError(
                        f"Failed to load model {model} with backend {backend}"
                    )

            except Exception as e:
                logger.error(f"Model loading failed: {e}")
                fallback_from = backend
                model_results, effective_backend = _run_fallback(
                    remaining_for_model,
                    clean_reference_entries,
                    backend,
                    threshold,
                    scored_threshold,
                )
            else:
                try:
                    if use_batch_optimization and len(remaining_for_model) > 10:
                        results = optimized_batch_matching(
                            remaining_for_model,
                            clean_reference_entries,
                            model_obj,
                            backend,
                            scored_threshold,
                        )
                        acceleration = "Batch-Optimized"
                    else:
                        results = standard_matching(
                            remaining_for_model,
                            clean_reference_entries,
                            model_obj,
                            backend,
                            scored_threshold,
                        )
                        acceleration = "Standard"

                    model_results = pd.DataFrame(results)

                except Exception as e:
                    logger.error(f"Processing failed: {e}")
                    fallback_from = backend
                    model_results, effective_backend = _run_fallback(
                        remaining_for_model,
                        clean_reference_entries,
                        backend,
                        threshold,
                        scored_threshold,
                    )

        final_results = _reconstruct_results_with_exact_and_model_matches(
            exact_matches,
            model_results,
            remaining_indices_for_model,
            original_indices,
            dirty_entries,
            nan_mask,
            verbose or _VERBOSE_MODE,
        )

        processing_time = time.time() - start_time
        final_results.attrs.update(
            {
                "acceleration": (
                    acceleration if "acceleration" in locals() else "API/RapidFuzz"
                ),
                "backend": backend,
                "effective_backend": effective_backend,
                # the backend that gave out, or None when the selected one ran
                "fallback_from": fallback_from,
                "processing_time": processing_time,
                "model": model,
                "nan_entries_count": sum(nan_mask),
                "exact_matches_count": len(exact_matches),
                "processed_entries_count": len(remaining_for_model),
            }
        )

        if verbose or _VERBOSE_MODE:
            acceleration_str = (
                acceleration if "acceleration" in locals() else "API/RapidFuzz"
            )
            logger.info(
                f"Processing completed in {processing_time:.2f} seconds using {acceleration_str}"
            )
            if sum(nan_mask) > 0:
                logger.info(f"Preserved {sum(nan_mask)} NaN entries in results")
            if len(exact_matches) > 0:
                logger.info(f"Found {len(exact_matches)} exact matches (score=1.0)")
        if return_model_attrs:
            final_results["alethia_method"] = (
                model if fallback_from is None else effective_backend
            )
            final_results["alethia_backend"] = effective_backend
        if remove_identical_hits:
            final_results = final_results[
                final_results.given_entity != final_results.alethia_prediction
            ]
        if drop_duplicates:
            final_results = final_results.drop_duplicates()
        return final_results

    finally:
        if not old_verbose:
            set_verbose(False)


def _reconstruct_results_with_exact_matches(
    exact_matches: dict[int, dict[str, Any]],
    original_entries: list[str],
    nan_mask: list[bool],
    verbose: bool = False,
) -> pd.DataFrame:
    """Splice exact matches and NaNs back into the caller's ordering."""
    full_results = []

    for i, entry in enumerate(original_entries):
        if nan_mask[i]:
            full_results.append(_no_match_row(entry))
        elif i in exact_matches:
            full_results.append(exact_matches[i])
        else:
            full_results.append(
                {
                    "given_entity": entry,
                    "alethia_prediction": entry,
                    "alethia_score": 1.0,
                }
            )

    if verbose:
        logger.info(f"Reconstructed {len(full_results)} results (exact matches only)")

    return pd.DataFrame(full_results)


def _reconstruct_results_with_exact_and_model_matches(
    exact_matches: dict[int, dict[str, Any]],
    model_results: pd.DataFrame,
    remaining_indices_for_model: list[int],
    original_indices: list[int],
    original_entries: list[str],
    nan_mask: list[bool],
    verbose: bool = False,
) -> pd.DataFrame:
    """Splice exact matches, model results and NaNs back into the caller's ordering."""
    full_results = []
    model_idx = 0

    for i, entry in enumerate(original_entries):
        if nan_mask[i]:
            full_results.append(_no_match_row(entry))
        elif i in exact_matches:
            full_results.append(exact_matches[i])
        else:
            if model_idx < len(model_results):
                result_row = model_results.iloc[model_idx].to_dict()
                result_row["given_entity"] = entry
                full_results.append(result_row)
                model_idx += 1
            else:
                full_results.append(
                    {
                        "given_entity": entry,
                        "alethia_prediction": entry,
                        "alethia_score": 1.0,
                    }
                )

    if verbose:
        exact_count = len(exact_matches)
        model_count = len(model_results)
        nan_count = sum(nan_mask)
        total_count = len(full_results)
        logger.info(
            f"Reconstructed {total_count} results: {exact_count} exact + {model_count} model + {nan_count} NaN"
        )

    return pd.DataFrame(full_results)


def _is_nan_entry(entry) -> bool:
    """Whether an entry counts as null under the package's null vocabulary."""
    if entry is None:
        return True
    if pd.isna(entry):
        return True
    if isinstance(entry, str):
        entry_lower = entry.lower().strip()
        if entry_lower in ["nan", "null", "none", "", "na", "n/a"]:
            return True
    return False


def _preprocess_entries_with_nans(entries: list[str], verbose: bool = False) -> tuple:
    """Strip NaN entries; returns (kept, nan_mask, original_indices)."""
    processed_entries = []
    nan_mask = []
    original_indices = []

    for i, entry in enumerate(entries):
        if _is_nan_entry(entry):
            nan_mask.append(True)
        else:
            nan_mask.append(False)
            processed_entries.append(entry)
            original_indices.append(i)

    if verbose:
        nan_count = sum(nan_mask)
        valid_count = len(processed_entries)
        total_count = len(entries)
        logger.info(
            f"Preprocessing: {valid_count} valid entries, {nan_count} NaN entries out of {total_count} total"
        )

    return processed_entries, nan_mask, original_indices


def _filter_nan_entries(entries: list[str], verbose: bool = False) -> list[str]:
    """Drop NaN entries from a reference list."""
    filtered_entries = [entry for entry in entries if not _is_nan_entry(entry)]

    if verbose:
        original_count = len(entries)
        filtered_count = len(filtered_entries)
        nan_count = original_count - filtered_count
        if nan_count > 0:
            logger.info(
                f"Filtered {nan_count} NaN entries from reference list ({filtered_count} remaining)"
            )

    return filtered_entries


def _no_match_frame(entries: list[str], *, acceleration: str) -> pd.DataFrame:
    """One no-match row per entry, with the run metadata a real match would carry."""
    df = pd.DataFrame(
        [_no_match_row(entry) for entry in entries],
        columns=["given_entity", "alethia_prediction", "alethia_score"],
    )
    df.attrs.update(
        {
            "acceleration": acceleration,
            "backend": "none",
            "effective_backend": "none",
            "fallback_from": None,
            "processing_time": 0.0,
            "model": "none",
            "nan_entries_count": sum(_is_nan_entry(e) for e in entries),
            "processed_entries_count": sum(not _is_nan_entry(e) for e in entries),
        }
    )
    return df


def _create_nan_only_results(entries: list[str]) -> pd.DataFrame:
    """Results frame for input where every entry is NaN."""
    return _no_match_frame(entries, acceleration="NaN-only")


def _create_no_match_results(entries: list[str]) -> pd.DataFrame:
    """Results frame for a run with no reference entries to match against."""
    return _no_match_frame(entries, acceleration="No-reference")


def enable_debug_logging():
    """Enable debug level logging"""
    set_verbose(True, "DEBUG")


def enable_info_logging():
    """Enable info level logging"""
    set_verbose(True, "INFO")


def disable_verbose_logging():
    """Disable verbose logging (return to minimal mode)"""
    set_verbose(False)


def _convert_memory_to_gb(memory_str):
    """Helper function to convert memory strings to GB float values"""
    if not memory_str or memory_str == "Unknown":
        return None

    try:
        memory_str = str(memory_str).upper()
        if "GB" in memory_str:
            return float(memory_str.replace("GB", ""))
        elif "MB" in memory_str:
            return round(float(memory_str.replace("MB", "")) / 1024, 2)
        else:
            return float(memory_str)
    except (ValueError, TypeError):
        return None


def get_available_models(
    backend: str = "all",
    include_api: bool = True,
    verbose: bool = False,
    include_details: bool = True,
    use_mteb_data: bool = True,
    sort_by: str = "performance",
) -> dict[str, list[str] | pd.DataFrame]:
    """List embedding models known to the installed backends, as a DataFrame."""
    from .models import (
        classify_embedding_models,
        get_detailed_model_info,
        load_mteb_dashboard_data,
    )

    available_models = {}

    mteb_df = pd.DataFrame()
    if use_mteb_data:
        try:
            mteb_df = load_mteb_dashboard_data()
            if verbose and not mteb_df.empty:
                print(
                    f"[ok] Loaded {len(mteb_df)} HuggingFace models from MTEB dashboard"
                )
        except Exception as e:
            if verbose:
                print(f"[!] Could not load MTEB data: {e}")

    if backend in ["all", "sentence-transformers"] and SENTENCE_TRANSFORMERS_AVAILABLE:
        model_details = get_detailed_model_info()
        classifications = classify_embedding_models()

        st_model_names = []
        for category_info in classifications.values():
            st_model_names.extend(category_info["models"])

        seen = set()
        st_model_names = [x for x in st_model_names if not (x in seen or seen.add(x))]

        if not mteb_df.empty:
            mteb_models = mteb_df["clean_model_name"].tolist()
            for mteb_model in mteb_models:
                if mteb_model not in st_model_names:
                    st_model_names.append(mteb_model)

        if include_details:
            st_data = []
            if verbose:
                print("Gathering Sentence Transformers model details...")

            iterator = (
                tqdm(st_model_names, desc="Getting model info")
                if verbose
                else st_model_names
            )

            for model_name in iterator:
                model_info = {
                    "model": model_name,
                    "backend": "sentence-transformers",
                    "available": True,
                }

                curated_info = model_details.get(model_name, {})
                if curated_info:
                    model_info.update(
                        {
                            "dimensions": curated_info.get("dimensions"),
                            "size_in_GB": _convert_memory_to_gb(
                                curated_info.get("estimated_memory")
                            ),
                            "estimated_params": curated_info.get("estimated_params"),
                            "organization": curated_info.get("organization"),
                            "size_category": curated_info.get("size_category"),
                            "best_use_case": curated_info.get("best_use_case"),
                            "data_source": "Curated",
                        }
                    )

                if not mteb_df.empty:
                    mteb_match = mteb_df[mteb_df["clean_model_name"] == model_name]
                    if not mteb_match.empty:
                        mteb_row = mteb_match.iloc[0]
                        model_info.update(
                            {
                                "mteb_rank": _safe_int_convert(
                                    mteb_row["Rank (Borda)"]
                                ),
                                "mteb_overall_score": _safe_float_convert(
                                    mteb_row["Mean (Task)"]
                                ),
                                "dimensions": _safe_int_convert(
                                    mteb_row["Embedding Dimensions"]
                                )
                                or model_info.get("dimensions"),
                                "size_in_GB": (
                                    mteb_row["memory_gb"]
                                    if pd.notna(mteb_row["memory_gb"])
                                    else model_info.get("size_in_GB")
                                ),
                                "max_seq_length": _safe_int_convert(
                                    mteb_row["Max Tokens"]
                                ),
                                "parameters": mteb_row["clean_parameters"],
                                "data_source": (
                                    "MTEB"
                                    if model_info.get("data_source") != "Curated"
                                    else "MTEB+Curated"
                                ),
                                "retrieval_score": _safe_float_convert(
                                    mteb_row["Retrieval"]
                                ),
                                "classification_score": _safe_float_convert(
                                    mteb_row["Classification"]
                                ),
                                "clustering_score": _safe_float_convert(
                                    mteb_row["Clustering"]
                                ),
                                "sts_score": _safe_float_convert(mteb_row["STS"]),
                                "reranking_score": _safe_float_convert(
                                    mteb_row["Reranking"]
                                ),
                            }
                        )

                for category, cat_info in classifications.items():
                    if model_name in cat_info["models"]:
                        model_info["category"] = category
                        model_info["category_description"] = cat_info["description"]
                        break
                else:
                    model_info["category"] = "uncategorized"

                st_data.append(model_info)

            st_df = pd.DataFrame(st_data)

            st_df = _sort_dataframe(st_df, sort_by)

            available_models["sentence-transformers"] = st_df
        else:
            available_models["sentence-transformers"] = st_model_names

        if verbose:
            count = len(st_model_names)
            mteb_count = len(mteb_df) if not mteb_df.empty else 0
            curated_count = len([m for m in st_model_names if m in model_details])

            print(f"[ok] Sentence Transformers: {count} models available")
            print(f"   * {mteb_count} from MTEB dashboard")
            print(f"   * {curated_count} from curated database")
            print(f"   * Sorted by: {sort_by}")

            if include_details and not st_df.empty:
                print("   Top models by current sorting:")
                top_models = st_df.head(3)
                for _, row in top_models.iterrows():
                    info_parts = []
                    if pd.notna(row.get("mteb_rank")):
                        info_parts.append(f"MTEB #{int(row['mteb_rank'])}")
                    if pd.notna(row.get("mteb_overall_score")):
                        info_parts.append(f"Score {row['mteb_overall_score']:.1f}")
                    if pd.notna(row.get("dimensions")):
                        info_parts.append(f"{int(row['dimensions'])}D")
                    if pd.notna(row.get("size_in_GB")) and isinstance(
                        row["size_in_GB"], (int, float)
                    ):
                        info_parts.append(f"{row['size_in_GB']:.2f}GB")

                    info_str = f"({', '.join(info_parts)})" if info_parts else ""
                    print(f"     {row['model']} {info_str}")

    if backend in ["all", "fastembed"] and FASTEMBED_AVAILABLE:
        try:
            from fastembed import TextEmbedding

            supported_models_raw = TextEmbedding.list_supported_models()

            if include_details:
                fastembed_df = pd.DataFrame(supported_models_raw)
                fastembed_df = fastembed_df.drop(
                    columns=[
                        "sources",
                        "tasks",
                        "description",
                        "license",
                        "model_file",
                        "additional_files",
                    ],
                    errors="ignore",
                )

                if sort_by == "size":
                    fastembed_df = fastembed_df.sort_values(
                        "size_in_GB", na_position="last"
                    )
                elif sort_by == "name":
                    fastembed_df = fastembed_df.sort_values("model")
                elif sort_by == "dimensions":
                    dim_col = "dim" if "dim" in fastembed_df.columns else "dimensions"
                    if dim_col in fastembed_df.columns:
                        fastembed_df = fastembed_df.sort_values(
                            dim_col, na_position="last"
                        )
                else:
                    fastembed_df = fastembed_df.sort_values(
                        "size_in_GB", na_position="last"
                    )

                fastembed_df = fastembed_df.reset_index(drop=True)
                available_models["fastembed"] = fastembed_df
            else:
                fastembed_models = [model["model"] for model in supported_models_raw]
                available_models["fastembed"] = fastembed_models

            if verbose:
                count = len(supported_models_raw)
                print(f"[ok] FastEmbed: {count} models available")
                if include_details:
                    print("   Sample models with details:")
                    sample_df = available_models["fastembed"].head(3)
                    for _, row in sample_df.iterrows():
                        dims = row.get("dim", row.get("dimensions", "Unknown"))
                        size = row.get("size_in_GB", "Unknown")
                        print(f"     {row['model']}: {dims}D, {size}GB")
                    print(f"     ... and {count - 3} more")

        except Exception as e:
            if verbose:
                print(f"[x] Could not retrieve FastEmbed models: {e}")
            available_models["fastembed"] = (
                [] if not include_details else pd.DataFrame()
            )

    if backend in ["all", "openai"] and include_api:
        if OPENAI_AVAILABLE:
            openai_data = [
                {
                    "model": "text-embedding-ada-002",
                    "dimensions": 1536,
                    "size_in_GB": "API",
                    "max_tokens": 8191,
                    "provider": "openai",
                },
                {
                    "model": "text-embedding-3-small",
                    "dimensions": 1536,
                    "size_in_GB": "API",
                    "max_tokens": 8191,
                    "provider": "openai",
                },
                {
                    "model": "text-embedding-3-large",
                    "dimensions": 3072,
                    "size_in_GB": "API",
                    "max_tokens": 8191,
                    "provider": "openai",
                },
            ]

            if include_details:
                openai_df = pd.DataFrame(openai_data)
                if sort_by == "name":
                    openai_df = openai_df.sort_values("model")
                elif sort_by == "dimensions":
                    openai_df = openai_df.sort_values("dimensions")
                available_models["openai"] = openai_df
            else:
                available_models["openai"] = [model["model"] for model in openai_data]

            if verbose:
                api_key_set = "[ok]" if os.getenv("OPENAI_API_KEY") else "[x]"
                print(f"{api_key_set} OpenAI: {len(openai_data)} models available")
        else:
            if verbose:
                print("[x] OpenAI: Not available (install with: pip install openai)")
            available_models["openai"] = [] if not include_details else pd.DataFrame()

    if backend in ["all", "gemini"] and include_api:
        if GEMINI_AVAILABLE:
            gemini_data = [
                {
                    "model": "models/embedding-001",
                    "dimensions": 768,
                    "size_in_GB": "API",
                    "max_tokens": 2048,
                    "provider": "gemini",
                },
                {
                    "model": "models/text-embedding-004",
                    "dimensions": 768,
                    "size_in_GB": "API",
                    "max_tokens": 2048,
                    "provider": "gemini",
                },
            ]

            if include_details:
                gemini_df = pd.DataFrame(gemini_data)
                if sort_by == "name":
                    gemini_df = gemini_df.sort_values("model")
                available_models["gemini"] = gemini_df
            else:
                available_models["gemini"] = [model["model"] for model in gemini_data]

            if verbose:
                api_key_set = "[ok]" if os.getenv("GEMINI_API_KEY") else "[x]"
                print(f"{api_key_set} Gemini: {len(gemini_data)} models available")
        else:
            if verbose:
                print(
                    "[x] Gemini: Not available (install with: pip install google-generativeai)"
                )
            available_models["gemini"] = [] if not include_details else pd.DataFrame()

    if backend in ["all", "rapidfuzz"]:
        if RAPIDFUZZ_AVAILABLE:
            rapidfuzz_data = [
                {
                    "method": "token_sort_ratio",
                    "description": "Token-based similarity with sorting",
                },
                {
                    "method": "token_set_ratio",
                    "description": "Token-based similarity with set operations",
                },
                {"method": "ratio", "description": "Standard Levenshtein ratio"},
                {"method": "partial_ratio", "description": "Partial string matching"},
            ]

            if include_details:
                rapidfuzz_df = pd.DataFrame(rapidfuzz_data)
                if sort_by == "name":
                    rapidfuzz_df = rapidfuzz_df.sort_values("method")
                available_models["rapidfuzz"] = rapidfuzz_df
            else:
                available_models["rapidfuzz"] = [
                    method["method"] for method in rapidfuzz_data
                ]

            if verbose:
                print(f"[ok] RapidFuzz: {len(rapidfuzz_data)} methods available")
        else:
            if verbose:
                print(
                    "[x] RapidFuzz: Not available (install with: pip install rapidfuzz)"
                )
            available_models["rapidfuzz"] = (
                [] if not include_details else pd.DataFrame()
            )

    if verbose:
        if include_details:
            total_models = sum(
                len(models) if isinstance(models, list) else len(models.index)
                for models in available_models.values()
                if len(models) > 0
            )
        else:
            total_models = sum(len(models) for models in available_models.values())

        available_backends = [
            k
            for k, v in available_models.items()
            if (isinstance(v, list) and v)
            or (isinstance(v, pd.DataFrame) and not v.empty)
        ]
        print(
            f"\nSummary: {total_models} total models across {len(available_backends)} backends"
        )
        print(f"Available backends: {', '.join(available_backends)}")

        print("\nSmart Recommendations:")
        if "sentence-transformers" in available_backends:
            if use_mteb_data and not mteb_df.empty and include_details:
                st_df = available_models["sentence-transformers"]

                if sort_by == "size":
                    lightweight = st_df[
                        st_df["size_in_GB"].notna() & (st_df["size_in_GB"] < 1.0)
                    ].head(3)
                    if not lightweight.empty:
                        print("   Smallest models:")
                        for _, row in lightweight.iterrows():
                            mem_str = (
                                f"{row['size_in_GB']:.2f}GB"
                                if pd.notna(row["size_in_GB"])
                                else "N/A"
                            )
                            print(f"     * {row['model']}: {mem_str}")
                elif sort_by == "performance":
                    top_mteb = st_df[st_df["mteb_rank"].notna()].head(3)
                    if not top_mteb.empty:
                        print("   Top MTEB performers:")
                        for _, row in top_mteb.iterrows():
                            mem_str = (
                                f"{row['size_in_GB']:.1f}GB"
                                if pd.notna(row["size_in_GB"])
                                and isinstance(row["size_in_GB"], (int, float))
                                else "N/A"
                            )
                            print(
                                f"     * {row['model']}: MTEB #{int(row['mteb_rank'])} ({row['mteb_overall_score']:.1f}), {mem_str}"
                            )
                else:
                    top_models = st_df.head(3)
                    print(f"   Top models by {sort_by}:")
                    for _, row in top_models.iterrows():
                        print(f"     * {row['model']}")

        if "openai" in available_backends and os.getenv("OPENAI_API_KEY"):
            print("   API option: text-embedding-3-small (no local compute needed)")

        if not available_backends:
            print("   * Install dependencies: pip install alethia[recommended]")

    return available_models


def _sort_dataframe(df: pd.DataFrame, sort_by: str) -> pd.DataFrame:
    """Sort by "performance", "size", "name" or "dimensions"."""
    if df.empty:
        return df

    if sort_by == "size":

        def size_sort_key(row):
            size = row.get("size_in_GB")
            if pd.isna(size) or not isinstance(size, (int, float)):
                return (1, 999)
            return (0, size)

        df["_size_sort_key"] = df.apply(size_sort_key, axis=1)
        df = df.sort_values("_size_sort_key").drop(columns=["_size_sort_key"])

    elif sort_by == "name":
        df = df.sort_values("model", na_position="last")

    elif sort_by == "dimensions":

        def dim_sort_key(row):
            dims = row.get("dimensions")
            if pd.isna(dims) or not isinstance(dims, (int, float)):
                return (1, 999)
            return (0, dims)

        df["_dim_sort_key"] = df.apply(dim_sort_key, axis=1)
        df = df.sort_values("_dim_sort_key").drop(columns=["_dim_sort_key"])

    elif sort_by == "performance":

        def performance_sort_key(row):
            if pd.notna(row.get("mteb_rank")):
                return (0, row["mteb_rank"])
            elif pd.notna(row.get("size_in_GB")) and isinstance(
                row["size_in_GB"], (int, float)
            ):
                return (1, row["size_in_GB"])
            else:
                return (2, 0)

        df["_perf_sort_key"] = df.apply(performance_sort_key, axis=1)
        df = df.sort_values("_perf_sort_key").drop(columns=["_perf_sort_key"])

    return df.reset_index(drop=True)


def _safe_float_convert(value):
    """Safely convert a value to float, handling 'Unknown' and other non-numeric values"""
    if pd.isna(value) or value == "Unknown" or value == "":
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _safe_int_convert(value):
    """Safely convert a value to int, handling 'Unknown' and other non-numeric values"""
    if pd.isna(value) or value == "Unknown" or value == "":
        return None
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return None


def get_models_by_size(
    backend: str = "sentence-transformers", ascending: bool = True
) -> pd.DataFrame:
    """Models whose size falls within ``[min_size_gb, max_size_gb]``."""
    models = get_available_models(
        backend=backend, include_details=True, sort_by="size", verbose=False
    )

    if backend in models and isinstance(models[backend], pd.DataFrame):
        df = models[backend].copy()
        if not ascending:
            df = df.iloc[::-1].reset_index(drop=True)
        return df

    return pd.DataFrame()


def get_smallest_models(max_size_gb: float = 1.0, top_n: int = 5) -> pd.DataFrame:
    """The ``top_n`` smallest models under ``max_size_gb``."""
    models = get_available_models(
        backend="sentence-transformers",
        include_details=True,
        sort_by="size",
        verbose=False,
    )

    if "sentence-transformers" in models:
        df = models["sentence-transformers"]
        small_models = df[
            (df["size_in_GB"].notna()) & (df["size_in_GB"] <= max_size_gb)
        ].head(top_n)

        return small_models

    return pd.DataFrame()
