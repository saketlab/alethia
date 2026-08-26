"""Top-level package for alethia.

Public names are resolved lazily (PEP 562), so importing this package does not pull in
torch, sentence-transformers, matplotlib or umap until a name backed by them is used.
"""

import sys
from importlib import import_module
from types import ModuleType
from typing import Any

__author__ = """Saket Choudhary"""
__email__ = "saketc@iitb.ac.in"
__version__ = "0.1.0"

# __all__ is derived from this
_EXPORTS: dict[str, str] = {
    "alethia": ".alethia",
    "load_sentence_transformer_model": ".alethia",
    "get_best_available_backend": ".alethia",
    "get_available_models": ".alethia",
    "check_optional_dependencies": ".alethia",
    "set_verbose": ".alethia",
    "enable_debug_logging": ".alethia",
    "enable_info_logging": ".alethia",
    "disable_verbose_logging": ".alethia",
    "print_resource_usage": ".alethia",
    "SENTENCE_TRANSFORMERS_AVAILABLE": ".alethia",
    "FASTEMBED_AVAILABLE": ".alethia",
    "RAPIDFUZZ_AVAILABLE": ".alethia",
    "OPENAI_AVAILABLE": ".alethia",
    "GEMINI_AVAILABLE": ".alethia",
    "Embedder": ".embedder",
    "as_embedder": ".embedder",
    "match_by_embeddings": ".embedder",
    "assess_models": ".assess",
    "assessment_table": ".assess",
    "AssessmentReport": ".assess",
    "ModelAssessment": ".assess",
    "generate_positive_pairs": ".assess",
    "make_dirty_variant": ".assess",
    "validate_assessor": ".assess",
    "LabeledDataset": ".assess",
    "ValidationResult": ".assess",
    "true_accuracy": ".assess",
    "cluster_entities": ".cluster",
    "mutual_nn_edges": ".cluster",
    "ClusterResult": ".cluster",
    "Edge": ".cluster",
    "classify_embedding_models": ".models",
    "create_recommendation_matrix": ".models",
    "get_model_recommendation": ".models",
    "print_model_classification_guide": ".models",
    "load_mteb_dashboard_data": ".models",
    "filter_huggingface_only": ".models",
    "get_embeddings": ".embeddings",
    "do_pca": ".stats",
    "do_umap": ".stats",
    "plot_embedding": ".stats",
    "plot_embedding_df": ".stats",
    "setup_matplotlib": ".utils",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve a public name on first access, then cache it in module globals."""
    try:
        module = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted([*globals(), *_EXPORTS])


class _Package(ModuleType):
    """Module type that keeps ``alethia.alethia`` bound to the matching function.

    The name is both the function and the module defining it, and Python binds the
    submodule onto its parent package as soon as anything imports it, landing in module
    globals ahead of ``__getattr__``. A property is a data descriptor, so it wins.
    """

    @property
    def alethia(self) -> Any:
        from .alethia import alethia

        return alethia


sys.modules[__name__].__class__ = _Package
