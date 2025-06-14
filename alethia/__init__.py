"""Top-level package for alethia."""

__author__ = """Saket Choudhary"""
__email__ = "saketkc@gmail.com"
__version__ = "0.1.0"

from .alethia import (FAISS_AVAILABLE, FASTEMBED_AVAILABLE, NUMBA_AVAILABLE,
                      RAPIDFUZZ_AVAILABLE, SENTENCE_TRANSFORMERS_AVAILABLE,
                      alethia, get_available_models,
                      get_best_available_backend,
                      load_sentence_transformer_model)
from .embeddings import get_embeddings
from .models import (classify_embedding_models, create_recommendation_matrix,
                     filter_huggingface_only,
                     get_model_recommendation, load_mteb_dashboard_data,
                     print_model_classification_guide)
from .stats import do_pca, do_umap, plot_embedding
from .utils import setup_matplotlib

__all__ = [
    # Core functionality
    "alethia",
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
