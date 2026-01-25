"""Alethia - Entity matching with embeddings."""

__author__ = "Saket Choudhary"
__email__ = "saketkc@gmail.com"
__version__ = "0.2.0"

from .alethia import (
    FASTEMBED_AVAILABLE,
    NUMBA_AVAILABLE,
    RAPIDFUZZ_AVAILABLE,
    SENTENCE_TRANSFORMERS_AVAILABLE,
    alethia,
    get_best_available_backend,
    get_fastembed_supported_models,
    is_model_fastembed_supported,
    load_fastembed_model,
    load_sentence_transformer_model,
)
from .embeddings import get_embeddings
from .models import filter_huggingface_only, load_mteb_dashboard_data
from .similarity import FastSimilarityMatcher
from .stats import do_pca, do_umap, plot_embedding, plot_embedding_df
from .utils import setup_matplotlib

__all__ = [
    "alethia",
    "load_sentence_transformer_model",
    "load_fastembed_model",
    "get_best_available_backend",
    "get_fastembed_supported_models",
    "is_model_fastembed_supported",
    "load_mteb_dashboard_data",
    "filter_huggingface_only",
    "get_embeddings",
    "do_pca",
    "do_umap",
    "plot_embedding",
    "plot_embedding_df",
    "setup_matplotlib",
    "FastSimilarityMatcher",
    "SENTENCE_TRANSFORMERS_AVAILABLE",
    "FASTEMBED_AVAILABLE",
    "RAPIDFUZZ_AVAILABLE",
    "NUMBA_AVAILABLE",
]
