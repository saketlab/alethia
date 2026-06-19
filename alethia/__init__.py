"""Model-agnostic entity matching and standardization with language-model embeddings."""

from importlib.metadata import PackageNotFoundError, version

__author__ = "Saket Choudhary"
__email__ = "saketc@iitb.ac.in"

try:
    __version__ = version("alethia")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

from .alethia import (
    FAISS_AVAILABLE,
    FASTEMBED_AVAILABLE,
    NUMBA_AVAILABLE,
    RAPIDFUZZ_AVAILABLE,
    SENTENCE_TRANSFORMERS_AVAILABLE,
    alethia,
    check_optional_dependencies,
    get_best_available_backend,
    load_sentence_transformer_model,
    set_verbose,
)

from .assess import (
    AssessmentReport,
    ModelAssessment,
    assess_models,
    assessment_table,
)

from .embedder import (
    CallableEmbedder,
    Embedder,
    as_embedder,
    match_by_embeddings,
)
from .embeddings import get_embeddings
from .cluster import (
    ClusterResult,
    Edge,
    cluster_entities,
    mutual_nn_edges,
)
from .stats import do_pca, do_umap, plot_embedding, plot_embedding_df
from .utils import setup_matplotlib

__all__ = [
    # Core matching
    "alethia",
    "load_sentence_transformer_model",
    "get_best_available_backend",
    "check_optional_dependencies",
    "set_verbose",
    # Model-agnostic embedding interface
    "Embedder",
    "CallableEmbedder",
    "as_embedder",
    "match_by_embeddings",
    # Entity clustering / deduplication
    "cluster_entities",
    "mutual_nn_edges",
    "ClusterResult",
    "Edge",
    # Label-free model assessment
    "assess_models",
    "assessment_table",
    "AssessmentReport",
    "ModelAssessment",
    # Embeddings & visualization
    "get_embeddings",
    "do_pca",
    "do_umap",
    "plot_embedding",
    "plot_embedding_df",
    "setup_matplotlib",
    # Dependency flags
    "SENTENCE_TRANSFORMERS_AVAILABLE",
    "FASTEMBED_AVAILABLE",
    "RAPIDFUZZ_AVAILABLE",
    "FAISS_AVAILABLE",
    "NUMBA_AVAILABLE",
]
