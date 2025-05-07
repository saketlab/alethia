"""Top-level package for alethia."""

__author__ = """Saket Choudhary"""
__email__ = "saketkc@gmail.com"
__version__ = "0.1.0"
from .alethia import alethia
from .alethia import load_sentence_transformer
from .alethia import get_embeddings

from .stats import do_pca
from .stats import do_umap
from .stats import plot_embedding
