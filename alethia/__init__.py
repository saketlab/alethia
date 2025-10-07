"""Top-level package for alethia."""

__author__ = """Saket Choudhary"""
__email__ = "saketkc@gmail.com"
__version__ = "0.1.0"

from .alethia import (
    FAISS_AVAILABLE,
    FASTEMBED_AVAILABLE,
    NUMBA_AVAILABLE,
    RAPIDFUZZ_AVAILABLE,
    SENTENCE_TRANSFORMERS_AVAILABLE,
    alethia,
    get_available_models,
    get_best_available_backend,
    load_sentence_transformer_model,
)
from .embeddings import get_embeddings
from .models import (
    classify_embedding_models,
    create_recommendation_matrix,
    filter_huggingface_only,
    get_model_recommendation,
    load_mteb_dashboard_data,
    print_model_classification_guide,
    get_best_cpu_model,
    get_cpu_optimized_models,
    get_model_for_cpu_setup,
    get_fastembed_supported_models,
    is_model_supported_by_fastembed,
    get_best_backend_for_model,
)
from .stats import do_pca, do_umap, plot_embedding, plot_embedding_df
from .utils import setup_matplotlib


def alethia_cpu_optimized(dirty_entries, reference_entries, **kwargs):
    """
    Run Alethia with optimal CPU settings using the best embedding model for CPU-only execution.

    This is a convenience function that automatically selects the best CPU-compatible model
    (intfloat/multilingual-e5-large-instruct) and ensures all computations run on CPU.

    Args:
        dirty_entries: List of incorrect entries to match
        reference_entries: List of reference entries to match against
        **kwargs: Additional arguments passed to the main alethia function

    Returns:
        DataFrame with matching results optimized for CPU execution

    Example:
        >>> from alethia import alethia_cpu_optimized
        >>> result = alethia_cpu_optimized(
        ...     dirty_entries=["NY", "LA", "Chiacgo"],
        ...     reference_entries=["New York", "Los Angeles", "Chicago"]
        ... )
    """
    # Set optimal CPU defaults
    cpu_defaults = {
        "model": "auto",  # Will auto-select best CPU model
        "backend": "sentence-transformers",  # Best for CPU
        "force_cpu": True,
        "use_batch_optimization": True,
        "verbose": kwargs.get("verbose", True),  # Enable progress by default
    }

    # Update with user-provided kwargs
    cpu_defaults.update(kwargs)

    return alethia(dirty_entries, reference_entries, **cpu_defaults)


__all__ = [
    # Core functionality
    "alethia",
    "alethia_cpu_optimized",
    "load_sentence_transformer_model",
    "get_best_available_backend",
    "check_optional_dependencies",
    # Model classification and recommendations
    "classify_embedding_models",
    "get_detailed_model_info",
    "create_recommendation_matrix",
    "get_model_recommendation",
    "print_model_classification_guide",
    "load_mteb_dashboard_data",
    "filter_huggingface_only",
    "get_available_models",
    "print_model_recommendations",
    # CPU-optimized models
    "get_best_cpu_model",
    "get_cpu_optimized_models",
    "get_model_for_cpu_setup",
    "get_fastembed_supported_models",
    "is_model_supported_by_fastembed",
    "get_best_backend_for_model",
    # Convenience functions
    "get_lightweight_models",
    "get_balanced_models",
    "get_high_quality_models",
    "get_multilingual_models",
    # Utility functions
    "get_embeddings",
    "do_pca",
    "do_umap",
    "plot_embedding",
    "plot_embedding_df",
    "setup_matplotlib",
    "print_resource_usage",
    # Configuration and debugging
    "set_verbose",
    "enable_debug_logging",
    "enable_info_logging",
    "disable_verbose_logging",
    "check_installation",
    "quick_start",
    # Dependency flags
    "SENTENCE_TRANSFORMERS_AVAILABLE",
    "FASTEMBED_AVAILABLE",
    "RAPIDFUZZ_AVAILABLE",
    "FAISS_AVAILABLE",
    "NUMBA_AVAILABLE",
    "OPENAI_AVAILABLE",
    "GEMINI_AVAILABLE",
]
