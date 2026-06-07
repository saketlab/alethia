import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from .cpu_optimizations import (
    ClinicalCPUOptimizationConfig,
    get_cpu_optimized_model,
)
from .utils import (
    get_client,
    print_resource_usage,
    prompt_fuzzy_match,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_VERBOSE_MODE = False


def set_verbose(verbose: bool = True, level: str = "INFO"):
    """
    Set global verbose mode and logging level

    Args:
        verbose: Enable verbose logging
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    """
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


def check_optional_dependencies(verbose: bool = False):
    """
    Check for optional dependencies and return availability status

    Args:
        verbose: Whether to print detailed dependency information

    Returns:
        Dict[str, bool]: Dictionary mapping package names to availability status
    """
    dependencies = {
        "SENTENCE_TRANSFORMERS_AVAILABLE": False,
        "FASTEMBED_AVAILABLE": False,
        "RAPIDFUZZ_AVAILABLE": False,
        "FAISS_AVAILABLE": False,
        "NUMBA_AVAILABLE": False,
        "OPENAI_AVAILABLE": False,
        "GEMINI_AVAILABLE": False,
    }

    original_level = logger.level
    if verbose or _VERBOSE_MODE:
        logger.setLevel(logging.INFO)

    try:
        import torch
        from sentence_transformers import SentenceTransformer

        dependencies["SENTENCE_TRANSFORMERS_AVAILABLE"] = True
        if verbose or _VERBOSE_MODE:
            logger.info("✅ SentenceTransformers available")
        globals()["SentenceTransformer"] = SentenceTransformer
        globals()["torch"] = torch
    except ImportError:
        if verbose or _VERBOSE_MODE:
            logger.info("❌ SentenceTransformers not available")
        globals()["SentenceTransformer"] = None
        globals()["torch"] = None

    try:
        import fastembed

        dependencies["FASTEMBED_AVAILABLE"] = True
        if verbose or _VERBOSE_MODE:
            logger.info("✅ FastEmbed available")
    except ImportError:
        if verbose or _VERBOSE_MODE:
            logger.debug("❌ FastEmbed not available")

    try:
        from rapidfuzz import fuzz, process

        dependencies["RAPIDFUZZ_AVAILABLE"] = True
        if verbose or _VERBOSE_MODE:
            logger.debug("✅ RapidFuzz available")
        globals()["fuzz"] = fuzz
        globals()["process"] = process
    except ImportError:
        if verbose or _VERBOSE_MODE:
            logger.debug("❌ RapidFuzz not available")

    try:
        import faiss

        dependencies["FAISS_AVAILABLE"] = True
        if verbose or _VERBOSE_MODE:
            logger.info("✅ FAISS available")
    except ImportError:
        if verbose or _VERBOSE_MODE:
            logger.debug("❌ FAISS not available")

    try:
        from numba import jit

        dependencies["NUMBA_AVAILABLE"] = True
        if verbose or _VERBOSE_MODE:
            logger.info("✅ Numba available")
        globals()["jit"] = jit
    except ImportError:
        if verbose or _VERBOSE_MODE:
            logger.debug("❌ Numba not available (will use pure Python fallback)")

        def jit(nopython=True):
            def decorator(func):
                return func

            return decorator

        globals()["jit"] = jit

    try:
        from openai import OpenAI

        dependencies["OPENAI_AVAILABLE"] = True
        if verbose or _VERBOSE_MODE:
            logger.debug("✅ OpenAI available")
        globals()["OpenAI"] = OpenAI
    except ImportError:
        if verbose or _VERBOSE_MODE:
            logger.debug("❌ OpenAI not available")
        globals()["OpenAI"] = None

    try:
        import google.generativeai as genai

        dependencies["GEMINI_AVAILABLE"] = True
        if verbose or _VERBOSE_MODE:
            logger.debug("✅ Gemini (Google GenerativeAI) available")
        globals()["genai"] = genai
    except ImportError:
        if verbose or _VERBOSE_MODE:
            logger.debug("❌ Gemini (Google GenerativeAI) not available")
        globals()["genai"] = None

    logger.setLevel(original_level)

    unavailable_core = []
    if (
        not dependencies["SENTENCE_TRANSFORMERS_AVAILABLE"]
        and not dependencies["FASTEMBED_AVAILABLE"]
    ):
        unavailable_core.append("embedding models (sentence-transformers or fastembed)")
    if not dependencies["RAPIDFUZZ_AVAILABLE"]:
        unavailable_core.append("fuzzy matching (rapidfuzz)")

    if unavailable_core and not verbose and not _VERBOSE_MODE:
        print(f"⚠️  Missing optional dependencies: {', '.join(unavailable_core)}")
        print(
            "   Install with: pip install alethia[recommended] for full functionality"
        )

    return dependencies


DEPENDENCIES = check_optional_dependencies()
SENTENCE_TRANSFORMERS_AVAILABLE = DEPENDENCIES["SENTENCE_TRANSFORMERS_AVAILABLE"]
FASTEMBED_AVAILABLE = DEPENDENCIES["FASTEMBED_AVAILABLE"]
RAPIDFUZZ_AVAILABLE = DEPENDENCIES["RAPIDFUZZ_AVAILABLE"]
FAISS_AVAILABLE = DEPENDENCIES["FAISS_AVAILABLE"]
NUMBA_AVAILABLE = DEPENDENCIES["NUMBA_AVAILABLE"]
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


@jit(nopython=True)
def fast_cosine_similarity_matrix(A, B):
    """JIT-compiled cosine similarity matrix computation"""
    return np.dot(A, B.T)


@jit(nopython=True)
def fast_normalize_embeddings(embeddings):
    """JIT-compiled L2 normalization"""
    norms = np.sqrt(np.sum(embeddings**2, axis=1))
    return embeddings / norms.reshape(-1, 1)


def get_openai_embedding(client, text: str, model: str = "text-embedding-ada-002"):
    """Get embedding from OpenAI API"""
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding


def get_gemini_embedding(text: str, model_name: str = "models/embedding-001"):
    """Get embedding from Gemini API"""
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
            logger.info(f"✅ Successfully loaded {fastembed_model_name} with FastEmbed")
        return model

    except Exception as e:
        logger.error(f"Error loading FastEmbed model: {e}")
        return None


def load_sentence_transformer_model(
    model_name: str,
    force_cpu: bool = False,
    optimize_cpu: bool = True,
    optimization_config: Optional[ClinicalCPUOptimizationConfig] = None,
    **model_kwargs,
):
    """Load SentenceTransformer model with proper error handling"""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "SentenceTransformers not available. Install with: pip install sentence-transformers"
        )

    try:
        kwargs = dict(model_kwargs)
        trust_remote = kwargs.pop("trust_remote_code", True)

        if (force_cpu or not torch.cuda.is_available()) and optimize_cpu:
            optimized_model = get_cpu_optimized_model(
                model_name,
                config=optimization_config,
                model_kwargs={**kwargs, "trust_remote_code": trust_remote},
            )
            if optimized_model is not None:
                return optimized_model

        if force_cpu or not torch.cuda.is_available():
            device = "cpu"
            if _VERBOSE_MODE:
                logger.info(f"Loading {model_name} on CPU")
        else:
            device = "cuda"
            if _VERBOSE_MODE:
                logger.info(f"Loading {model_name} on GPU")

        model = SentenceTransformer(
            model_name,
            device=device,
            trust_remote_code=trust_remote,
            **kwargs,
        )
        if _VERBOSE_MODE:
            logger.info(f"✅ Successfully loaded {model_name}")
        return model

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            logger.warning("GPU memory error, trying CPU")
            try:
                model = SentenceTransformer(
                    model_name,
                    device="cpu",
                    trust_remote_code=trust_remote,
                    **kwargs,
                )
                if _VERBOSE_MODE:
                    logger.info(f"✅ Successfully loaded {model_name} on CPU")
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

    api_key = os.getenv("GEMINI_API_KEY")
    if api_key is None:
        raise ValueError("GEMINI_API_KEY not set in environment variables.")

    genai.configure(api_key=api_key)
    return genai


def get_best_available_backend(prefer_cpu: bool = False):
    """Get the best available backend"""
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


def _model_label(model) -> str:
    """A short string identifying a model spec (name, callable, or loaded object)."""
    if isinstance(model, str):
        return model
    return getattr(model, "__name__", type(model).__name__)


def run_openai_matching(
    dirty_entries: List[str],
    reference_entries: List[str],
    model_name: str = "text-embedding-ada-002",
    threshold: float = 0.7,
) -> pd.DataFrame:
    """Run OpenAI-based matching"""
    if not OPENAI_AVAILABLE:
        raise ImportError("OpenAI not available. Install with: pip install openai")

    client = setup_openai_client()

    if _VERBOSE_MODE:
        logger.info("Computing OpenAI reference embeddings...")
    reference_embeddings = {}

    iterator = (
        tqdm(reference_entries, desc="Reference embeddings")
        if (_VERBOSE_MODE or len(reference_entries) > 20)
        else reference_entries
    )
    for ref_entity in iterator:
        embedding = get_openai_embedding(client, ref_entity, model_name)
        reference_embeddings[ref_entity] = np.array(embedding)

    results = []
    if _VERBOSE_MODE:
        logger.info("Processing queries with OpenAI...")

    iterator = (
        tqdm(dirty_entries, desc="Processing queries")
        if (_VERBOSE_MODE or len(dirty_entries) > 20)
        else dirty_entries
    )
    for incorrect in iterator:
        if str(incorrect) == "nan":
            results.append(
                {
                    "given_entity": incorrect,
                    "alethia_prediction": np.nan,
                    "alethia_score": np.nan,
                }
            )
            continue

        query_embedding = np.array(get_openai_embedding(client, incorrect, model_name))

        similarities = {}
        for ref_entity, ref_embedding in reference_embeddings.items():
            similarity = cosine_similarity(query_embedding, ref_embedding)
            similarities[ref_entity] = similarity

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
            results.append(
                {
                    "given_entity": incorrect,
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
    """Run Gemini-based matching"""
    if not GEMINI_AVAILABLE:
        raise ImportError(
            "Google GenerativeAI not available. Install with: pip install google-generativeai"
        )

    setup_gemini_client()

    if _VERBOSE_MODE:
        logger.info("Computing Gemini reference embeddings...")
    reference_embeddings = {}

    iterator = (
        tqdm(reference_entries, desc="Reference embeddings")
        if (_VERBOSE_MODE or len(reference_entries) > 20)
        else reference_entries
    )
    for ref_entity in iterator:
        embedding = get_gemini_embedding(ref_entity, model_name)
        reference_embeddings[ref_entity] = np.array(embedding)

    results = []
    if _VERBOSE_MODE:
        logger.info("Processing queries with Gemini...")

    iterator = (
        tqdm(dirty_entries, desc="Processing queries")
        if (_VERBOSE_MODE or len(dirty_entries) > 20)
        else dirty_entries
    )
    for incorrect in iterator:
        if str(incorrect) == "nan":
            results.append(
                {
                    "given_entity": incorrect,
                    "alethia_prediction": np.nan,
                    "alethia_score": np.nan,
                }
            )
            continue

        query_embedding = np.array(get_gemini_embedding(incorrect, model_name))

        similarities = {}
        for ref_entity, ref_embedding in reference_embeddings.items():
            similarity = cosine_similarity(query_embedding, ref_embedding)
            similarities[ref_entity] = similarity

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
            results.append(
                {
                    "given_entity": incorrect,
                    "alethia_prediction": incorrect,
                    "alethia_score": 1.0,
                }
            )

    return pd.DataFrame(results)


def optimized_batch_matching(
    dirty_entries, reference_entries, model_obj, backend, threshold=0.7
):
    """Match dirty entries to references using a loaded embedding model.

    Thin wrapper over the unified embedding engine (:func:`alethia.embedder.match_by_embeddings`),
    kept for backward compatibility. ``model_obj`` is a loaded SentenceTransformer or
    FastEmbed model; ``backend`` selects how it is wrapped.
    """
    from .embedder import (
        FastEmbedEmbedder,
        SentenceTransformerEmbedder,
        match_by_embeddings,
    )

    if backend == "fastembed":
        embedder = FastEmbedEmbedder(model_obj)
    else:
        embedder = SentenceTransformerEmbedder(model_obj)
    return match_by_embeddings(dirty_entries, reference_entries, embedder)


# Retained for backward compatibility; the unified engine made the two paths identical.
standard_matching = optimized_batch_matching


def run_rapidfuzz_matching(
    dirty_entries: List[str], reference_entries: List[str]
) -> pd.DataFrame:
    """Run RapidFuzz-based matching"""
    if not RAPIDFUZZ_AVAILABLE:
        raise ImportError(
            "RapidFuzz not available. Install with: pip install rapidfuzz"
        )

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
            # No matches found (empty reference list)
            results.append(
                {
                    "given_entity": incorrect,
                    "alethia_prediction": incorrect,
                    "alethia_score": 1.0,
                }
            )

    return pd.DataFrame(results)


def _embedding_fallback(
    dirty_entries: List[str],
    reference_entries: List[str],
    failed_backend: str,
    threshold: float,
    verbose: bool,
) -> pd.DataFrame:
    """Match via the best available non-embedding engine when an embedder is unusable.

    Tries RapidFuzz, then OpenAI, then Gemini, skipping ``failed_backend``. The backend
    actually used is recorded in ``result.attrs['fallback_backend']``. Raises if no
    fallback is available.
    """
    chain = [
        (
            "rapidfuzz",
            RAPIDFUZZ_AVAILABLE,
            lambda: run_rapidfuzz_matching(dirty_entries, reference_entries),
        ),
        (
            "openai",
            OPENAI_AVAILABLE,
            lambda: run_openai_matching(
                dirty_entries, reference_entries, "text-embedding-ada-002", threshold
            ),
        ),
        (
            "gemini",
            GEMINI_AVAILABLE,
            lambda: run_gemini_matching(
                dirty_entries, reference_entries, "models/embedding-001", threshold
            ),
        ),
    ]
    for name, available, run in chain:
        if name != failed_backend and available:
            if verbose:
                logger.info(f"Falling back to {name}")
            df = run()
            df.attrs["fallback_backend"] = name
            return df
    raise RuntimeError(
        f"No matching backend available (embedder failed; backend={failed_backend!r})."
    )


def _find_exact_matches(
    dirty_entries: List[str],
    reference_entries: List[str],
    case_sensitive: bool = False,
    verbose: bool = False,
) -> tuple:
    """
    Find exact matches between dirty entries and reference entries

    Args:
        dirty_entries: List of entries to match
        reference_entries: List of reference entries
        case_sensitive: Whether to perform case-sensitive matching
        verbose: Whether to log matching info

    Returns:
        tuple: (exact_matches_dict, remaining_dirty_entries, remaining_indices)
    """
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
            # Keep NaN entries for later processing
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
            # Case-insensitive matching
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


def _merge_exact_and_model_results(
    exact_matches: Dict[int, Dict[str, Any]],
    model_results: pd.DataFrame,
    remaining_indices: List[int],
    original_entries: List[str],
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Merge exact match results with model-based results

    Args:
        exact_matches: Dictionary of exact matches by original index
        model_results: Results from model-based matching
        remaining_indices: Indices of entries that went through model matching
        original_entries: Original input entries
        verbose: Whether to log merging info

    Returns:
        pd.DataFrame: Combined results in original order
    """
    final_results = []
    model_idx = 0

    for i, original_entry in enumerate(original_entries):
        if i in exact_matches:
            # Use exact match result
            final_results.append(exact_matches[i])
        else:
            # Use model result
            if model_idx < len(model_results):
                result_row = model_results.iloc[model_idx].to_dict()
                result_row["given_entity"] = original_entry
                final_results.append(result_row)
                model_idx += 1
            else:
                # Fallback (shouldn't happen in normal operation)
                final_results.append(
                    {
                        "given_entity": original_entry,
                        "alethia_prediction": original_entry,
                        "alethia_score": 1.0,
                    }
                )

    if verbose:
        exact_count = len(exact_matches)
        model_count = len(model_results)
        total_count = len(final_results)
        logger.info(
            f"Merged results: {exact_count} exact matches + {model_count} model matches = {total_count} total"
        )

    return pd.DataFrame(final_results)


def alethia(
    dirty_entries: List[str],
    reference_entries: List[str],
    model: Union[str, Callable[[List[str]], Any], Any] = "rapidfuzz",
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
    api_key: str = "",
    **kwargs,
) -> pd.DataFrame:
    """
    Main Alethia function with exact match pre-filtering and optimizations

    Args:
        dirty_entries: List of incorrect entries
        reference_entries: List of reference entries
        model: An embedding model name, an ``embed_fn(list[str]) -> ndarray`` callable, a
            loaded model object (SentenceTransformer / FastEmbed), or one of the string
            keywords ``"rapidfuzz"``, ``"openai"``, ``"gemini"``.
        backend: Backend to use ('auto', 'sentence-transformers', 'fastembed', 'rapidfuzz', 'openai', 'gemini', 'instructor')
        force_cpu: Force CPU usage
        use_batch_optimization: Use batch optimization
        threshold: Similarity threshold
        verbose: Enable verbose logging and progress bars
        use_exact_matching: Enable exact match pre-filtering
        exact_match_case_sensitive: Whether exact matching should be case-sensitive
        return_model_attrs: Include model and backend columns in results (default: True)
        drop_duplicates: Remove duplicate rows from results (default: True)
        remove_identical_hits: Remove rows where prediction equals input - useful when dirty entries
            are in reference list and you don't want self-matches (default: False)
        api_key: API key for the instructor if required
        **kwargs: Additional arguments (model_name for API backends)

    Returns:
        DataFrame with results, preserving NaN entries and including exact matches
    """
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

        # Handle NaN entries first
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

        # EXACT MATCHING PHASE
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

            # Map exact matches back to original indices
            original_exact_matches = {}
            for proc_idx, match_result in exact_matches.items():
                original_idx = original_indices[proc_idx]
                original_exact_matches[original_idx] = match_result
            exact_matches = original_exact_matches

        # If all entries were exact matches, return early
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
                    "processing_time": processing_time,
                    "model": "exact",
                    "nan_entries_count": sum(nan_mask),
                    "exact_matches_count": len(exact_matches),
                    "processed_entries_count": 0,
                }
            )

            final_results["alethia_method"] = "exact"

            # Apply remove_identical_hits filter before returning
            if remove_identical_hits:
                final_results = final_results[
                    final_results.given_entity != final_results.alethia_prediction
                ]
                if verbose or _VERBOSE_MODE:
                    logger.info(
                        f"Filtered out {len(exact_matches) - len(final_results)} self-matches"
                    )

            if drop_duplicates:
                final_results = final_results.drop_duplicates()

            return final_results

        # MODEL-BASED MATCHING PHASE (for remaining entries)
        if verbose or _VERBOSE_MODE:
            logger.info(
                f"Processing {len(remaining_for_model)} entries through model matching"
            )

        # String keywords select the non-embedding engines. Everything else (a model name,
        # an embed_fn callable, or a loaded model object) uses the embedding engine.
        if model == "rapidfuzz" or backend == "rapidfuzz":
            model_results = run_rapidfuzz_matching(
                remaining_for_model, clean_reference_entries
            )
        elif model == "openai" or backend == "openai":
            model_name = kwargs.get("model_name", "text-embedding-ada-002")
            model_results = run_openai_matching(
                remaining_for_model, clean_reference_entries, model_name, threshold
            )
        elif model == "gemini" or backend == "gemini":
            model_name = kwargs.get("model_name", "models/embedding-001")
            model_results = run_gemini_matching(
                remaining_for_model, clean_reference_entries, model_name, threshold
            )
        elif backend == "instructor":
            if verbose or _VERBOSE_MODE:
                logger.info(f"Using instructor for fuzzy matching with model {model}")

            try:
                prompt_matcher = get_client(model_name=model, api_key=api_key)
            except Exception as e:
                logger.error(f"Instructor model loading failed: {e}")
                raise

            model_results_list = []
            for query in tqdm(remaining_for_model, desc="Instructor matching"):
                if query:
                    match_result = prompt_fuzzy_match(
                        prompt_matcher, query, clean_reference_entries
                    )
                    model_results_list.append(
                        {
                            "given_entity": query,
                            "alethia_prediction": match_result["text"],
                            "alethia_score": match_result["score"]
                            / 100.0,  # convert back to 0-1 scale
                        }
                    )

            model_results = pd.DataFrame(model_results_list)
        else:
            # --- Unified embedding engine ---------------------------------------------
            # A model name, a callable embed_fn, or a loaded model object all become an
            # Embedder; match_by_embeddings is the single matching primitive. If a named
            # model cannot be resolved, fall back to fuzzy/API matching as before.
            from .embedder import as_embedder, match_by_embeddings

            embedder = None
            try:
                embedder = as_embedder(model, backend=backend, force_cpu=force_cpu)
            except Exception as e:
                logger.error(f"Could not build embedder for {model!r}: {e}")
                model_results = _embedding_fallback(
                    remaining_for_model,
                    clean_reference_entries,
                    backend,
                    threshold,
                    verbose or _VERBOSE_MODE,
                )
                backend = model_results.attrs.get("fallback_backend", backend)

            if embedder is not None:
                try:
                    results = match_by_embeddings(
                        remaining_for_model, clean_reference_entries, embedder
                    )
                    model_results = pd.DataFrame(results)
                    backend = embedder.family
                    acceleration = "Embedding+Numba" if NUMBA_AVAILABLE else "Embedding"
                except Exception as e:
                    logger.error(f"Embedding matching failed: {e}")
                    model_results = _embedding_fallback(
                        remaining_for_model,
                        clean_reference_entries,
                        backend,
                        threshold,
                        verbose or _VERBOSE_MODE,
                    )
                    backend = model_results.attrs.get("fallback_backend", backend)

        # MERGE EXACT MATCHES WITH MODEL RESULTS
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
                "processing_time": processing_time,
                "model": _model_label(model),
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
            final_results["alethia_method"] = _model_label(model)
            final_results["alethia_backend"] = backend
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
    exact_matches: Dict[int, Dict[str, Any]],
    original_entries: List[str],
    nan_mask: List[bool],
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Reconstruct results when only exact matches were found

    Args:
        exact_matches: Dictionary of exact matches by original index
        original_entries: Original input entries
        nan_mask: Boolean mask indicating which entries were NaN
        verbose: Whether to log reconstruction info

    Returns:
        pd.DataFrame: Results with exact matches and NaN entries
    """
    full_results = []

    for i, entry in enumerate(original_entries):
        if nan_mask[i]:
            full_results.append(
                {
                    "given_entity": entry,
                    "alethia_prediction": np.nan,
                    "alethia_score": np.nan,
                }
            )
        elif i in exact_matches:
            full_results.append(exact_matches[i])
        else:
            # This shouldn't happen if exact matching is working correctly
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
    exact_matches: Dict[int, Dict[str, Any]],
    model_results: pd.DataFrame,
    remaining_indices_for_model: List[int],
    original_indices: List[int],
    original_entries: List[str],
    nan_mask: List[bool],
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Reconstruct full results combining exact matches, model results, and NaN entries

    Args:
        exact_matches: Dictionary of exact matches by original index
        model_results: Results from model-based matching
        remaining_indices_for_model: Indices within processed entries that went to model
        original_indices: Mapping from processed to original indices
        original_entries: Original input entries
        nan_mask: Boolean mask indicating which entries were NaN
        verbose: Whether to log reconstruction info

    Returns:
        pd.DataFrame: Complete results with all matches preserved
    """
    full_results = []
    model_idx = 0

    for i, entry in enumerate(original_entries):
        if nan_mask[i]:
            # NaN entry
            full_results.append(
                {
                    "given_entity": entry,
                    "alethia_prediction": np.nan,
                    "alethia_score": np.nan,
                }
            )
        elif i in exact_matches:
            # Exact match
            full_results.append(exact_matches[i])
        else:
            # Model-based match
            if model_idx < len(model_results):
                result_row = model_results.iloc[model_idx].to_dict()
                result_row["given_entity"] = entry
                full_results.append(result_row)
                model_idx += 1
            else:
                # Fallback
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
    """
    Check if an entry should be considered NaN/null

    Args:
        entry: Entry to check

    Returns:
        bool: True if entry is NaN/null
    """
    if entry is None:
        return True
    if pd.isna(entry):
        return True
    if isinstance(entry, str):
        entry_lower = entry.lower().strip()
        if entry_lower in ["nan", "null", "none", "", "na", "n/a"]:
            return True
    return False


def _preprocess_entries_with_nans(entries: List[str], verbose: bool = False) -> tuple:
    """
    Preprocess entries to separate valid entries from NaN entries

    Args:
        entries: List of entries to process
        verbose: Whether to log preprocessing info

    Returns:
        tuple: (processed_entries, nan_mask, original_indices)
    """
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


def _filter_nan_entries(entries: List[str], verbose: bool = False) -> List[str]:
    """
    Filter out NaN entries from reference list

    Args:
        entries: List of entries to filter
        verbose: Whether to log filtering info

    Returns:
        List[str]: Filtered entries without NaNs
    """
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


def _reconstruct_results_with_nans(
    processed_results: pd.DataFrame,
    original_entries: List[str],
    nan_mask: List[bool],
    original_indices: List[int],
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Reconstruct full results DataFrame including NaN entries

    Args:
        processed_results: Results from processing valid entries
        original_entries: Original input entries
        nan_mask: Boolean mask indicating which entries were NaN
        original_indices: Indices of valid entries in original list
        verbose: Whether to log reconstruction info

    Returns:
        pd.DataFrame: Complete results with NaN entries preserved
    """
    full_results = []
    processed_idx = 0

    for i, entry in enumerate(original_entries):
        if nan_mask[i]:
            full_results.append(
                {
                    "given_entity": entry,
                    "alethia_prediction": np.nan,
                    "alethia_score": np.nan,
                }
            )
        else:
            if processed_idx < len(processed_results):
                result_row = processed_results.iloc[processed_idx].to_dict()
                result_row["given_entity"] = entry
                full_results.append(result_row)
                processed_idx += 1
            else:
                full_results.append(
                    {
                        "given_entity": entry,
                        "alethia_prediction": entry,
                        "alethia_score": 1.0,
                    }
                )

    if verbose:
        logger.info(f"Reconstructed {len(full_results)} total results")

    return pd.DataFrame(full_results)


def _create_nan_only_results(entries: List[str]) -> pd.DataFrame:
    """
    Create results DataFrame when all entries are NaN

    Args:
        entries: Original entries (all NaN)

    Returns:
        pd.DataFrame: Results with all NaN predictions
    """
    results = []
    for entry in entries:
        results.append(
            {
                "given_entity": entry,
                "alethia_prediction": np.nan,
                "alethia_score": np.nan,
            }
        )

    df = pd.DataFrame(results)
    df.attrs.update(
        {
            "acceleration": "NaN-only",
            "backend": "none",
            "processing_time": 0.0,
            "model": "none",
            "nan_entries_count": len(entries),
            "processed_entries_count": 0,
        }
    )

    return df


def _create_no_match_results(entries: List[str]) -> pd.DataFrame:
    """
    Create results DataFrame when no reference entries are available

    Args:
        entries: Original entries

    Returns:
        pd.DataFrame: Results with no changes (identity mapping)
    """
    results = []
    for entry in entries:
        if _is_nan_entry(entry):
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
    df.attrs.update(
        {
            "acceleration": "No-reference",
            "backend": "none",
            "processing_time": 0.0,
            "model": "none",
            "nan_entries_count": sum(_is_nan_entry(e) for e in entries),
            "processed_entries_count": sum(not _is_nan_entry(e) for e in entries),
        }
    )

    return df


def enable_debug_logging():
    """Enable debug level logging"""
    set_verbose(True, "DEBUG")


def enable_info_logging():
    """Enable info level logging"""
    set_verbose(True, "INFO")


def disable_verbose_logging():
    """Disable verbose logging (return to minimal mode)"""
    set_verbose(False)
