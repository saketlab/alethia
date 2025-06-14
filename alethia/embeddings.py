import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
from tqdm import tqdm

# Import your existing dependencies from alethia.py
try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

try:
    import fastembed

    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False

logger = logging.getLogger(__name__)


def get_embeddings(
    texts: Union[str, List[str]],
    model_type: str = "sentence-transformer",
    model: Optional[Union[str, SentenceTransformer]] = None,
    client: Optional[Any] = None,
    api_key: Optional[str] = None,
    return_labels: bool = False,
    show_progress: bool = False,
) -> Union[List[float], np.ndarray, Dict[str, List[float]]]:
    """
    Get embeddings for text(s) using various embedding models.
    Enhanced version with better integration with alethia backends.
    """
    # Validate and prepare inputs
    if isinstance(texts, str):
        texts_list = [texts]
    else:
        texts_list = texts

    is_single_input = isinstance(texts, str)

    # Generate embeddings based on model type
    if model_type.lower() == "openai":
        from openai import OpenAI

        if client is None and api_key is None:
            raise ValueError(
                "Either client or api_key must be provided for OpenAI embeddings"
            )

        if client is None:
            client = OpenAI(api_key=api_key)

        model_name = model or "text-embedding-ada-002"

        if is_single_input:
            response = client.embeddings.create(input=texts_list, model=model_name)
            return response.data[0].embedding
        else:
            response = client.embeddings.create(input=texts_list, model=model_name)
            return [item.embedding for item in response.data]

    elif model_type.lower() == "google":
        import genai

        if model is None:
            raise ValueError("Model name must be provided for Google embeddings")

        if is_single_input:
            return genai.embed_content(model=model, content=texts)["embedding"]
        else:
            return {
                text: genai.embed_content(model=model, content=text)["embedding"]
                for text in texts_list
            }

    elif model_type.lower() == "sentence-transformer":
        if model is None:
            raise ValueError("SentenceTransformer model object must be provided")

        if not isinstance(model, SentenceTransformer):
            raise ValueError(
                "For sentence-transformer type, model must be a SentenceTransformer object"
            )

        if show_progress:
            embeddings = np.stack(
                [
                    model.encode(text)
                    for text in tqdm(texts_list, desc="Generating Embeddings")
                ]
            )
        else:
            embeddings = np.stack([model.encode(text) for text in texts_list])

        if return_labels:
            labels_array = np.array(texts_list).reshape(-1, 1)
            embeddings = np.hstack([embeddings, labels_array])

        if is_single_input and not return_labels:
            return embeddings[0]
        else:
            return embeddings

    elif model_type.lower() == "fastembed":
        # Add FastEmbed support
        if not FASTEMBED_AVAILABLE:
            raise ImportError(
                "FastEmbed not available. Install with: pip install fastembed"
            )

        if model is None:
            raise ValueError("FastEmbed model object must be provided")

        # Generate embeddings using FastEmbed
        if show_progress:
            embeddings = []
            for text in tqdm(texts_list, desc="Generating FastEmbed Embeddings"):
                embedding = next(model.embed([text]))
                embeddings.append(embedding)
            embeddings = np.stack(embeddings)
        else:
            embeddings = np.array(list(model.embed(texts_list)))

        if return_labels:
            labels_array = np.array(texts_list).reshape(-1, 1)
            embeddings = np.hstack([embeddings, labels_array])

        if is_single_input and not return_labels:
            return embeddings[0]
        else:
            return embeddings

    else:
        raise ValueError(
            "model_type must be one of: 'openai', 'google', 'sentence-transformer', or 'fastembed'"
        )


def load_embedding_model(
    model_name: str = "BAAI/bge-small-en",
    backend: str = "auto",
    force_cpu: bool = True,
    **kwargs,
) -> tuple[Any, str]:
    """
    Load embedding model using alethia's backend selection logic.

    Args:
        model_name: Name of the model to load
        backend: Backend to use ('auto', 'sentence-transformers', 'fastembed')
        force_cpu: Force CPU usage

    Returns:
        Tuple of (model_object, actual_backend_used)
    """

    def get_best_available_backend(prefer_cpu: bool = False):
        if prefer_cpu and FASTEMBED_AVAILABLE:
            return "fastembed"
        elif SENTENCE_TRANSFORMERS_AVAILABLE:
            return "sentence-transformers"
        elif FASTEMBED_AVAILABLE:
            return "fastembed"
        else:
            raise ImportError("No embedding backend available")

    def load_fastembed_model(model_name: str):
        if not FASTEMBED_AVAILABLE:
            raise ImportError("FastEmbed not available")

        from fastembed import TextEmbedding

        model_mapping = {
            "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
            "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
        }

        fastembed_model_name = model_mapping.get(model_name, model_name)
        model = TextEmbedding(model_name=fastembed_model_name)
        logger.info(f"✅ Successfully loaded {fastembed_model_name} with FastEmbed")
        return model

    def load_sentence_transformer_model(model_name: str, force_cpu: bool = False):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("SentenceTransformers not available")

        import torch

        if force_cpu or not torch.cuda.is_available():
            device = "cpu"
            logger.info(f"Loading {model_name} on CPU")
        else:
            device = "cuda"
            logger.info(f"Loading {model_name} on GPU")

        try:
            model = SentenceTransformer(
                model_name, device=device, trust_remote_code=True
            )
            logger.info(f"✅ Successfully loaded {model_name}")
            return model
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) and not force_cpu:
                logger.warning("GPU memory error, trying CPU")
                model = SentenceTransformer(
                    model_name, device="cpu", trust_remote_code=True
                )
                logger.info(f"✅ Successfully loaded {model_name} on CPU")
                return model
            else:
                raise

    # Determine backend
    if backend == "auto":
        backend = get_best_available_backend(prefer_cpu=force_cpu)
        logger.info(f"Auto-selected backend: {backend}")

    # Load model
    if backend == "fastembed":
        model_obj = load_fastembed_model(model_name)
        return model_obj, "fastembed"
    elif backend == "sentence-transformers":
        model_obj = load_sentence_transformer_model(model_name, force_cpu=force_cpu)
        return model_obj, "sentence-transformers"
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def calculate_embeddings_batch(
    texts: List[str],
    model_name: str = "BAAI/bge-small-en",
    backend: str = "auto",
    force_cpu: bool = True,
    batch_size: int = 64,
    show_progress: bool = True,
    return_model: bool = False,
    **kwargs,
) -> Union[np.ndarray, tuple[np.ndarray, Any, str]]:
    """
    Calculate embeddings for a batch of texts using alethia's infrastructure.

    Args:
        texts: List of texts to embed
        model_name: Name of the model to use
        backend: Backend preference
        force_cpu: Force CPU usage
        batch_size: Batch size for processing
        show_progress: Show progress bar
        return_model: Whether to return the loaded model object

    Returns:
        Embeddings array, optionally with model object and backend info
    """
    # Load model
    model_obj, actual_backend = load_embedding_model(
        model_name=model_name, backend=backend, force_cpu=force_cpu, **kwargs
    )

    # Calculate embeddings using your get_embeddings function
    if actual_backend == "fastembed":
        embeddings = get_embeddings(
            texts=texts,
            model_type="fastembed",
            model=model_obj,
            show_progress=show_progress,
        )
    else:  # sentence-transformers
        embeddings = get_embeddings(
            texts=texts,
            model_type="sentence-transformer",
            model=model_obj,
            show_progress=show_progress,
        )

    if return_model:
        return embeddings, model_obj, actual_backend
    else:
        return embeddings


# Example usage functions
def example_single_text():
    """Example: Calculate embedding for a single text"""
    text = "This is a sample text for embedding"

    # Load model
    model_obj, backend = load_embedding_model(
        model_name="all-MiniLM-L6-v2", backend="auto", force_cpu=True
    )

    # Get embedding
    if backend == "fastembed":
        embedding = get_embeddings(texts=text, model_type="fastembed", model=model_obj)
    else:
        embedding = get_embeddings(
            texts=text, model_type="sentence-transformer", model=model_obj
        )

    print(f"Embedding shape: {embedding.shape}")
    print(f"First 5 values: {embedding[:5]}")


def example_batch_texts():
    """Example: Calculate embeddings for multiple texts"""
    texts = [
        "Machine learning is fascinating",
        "Deep learning uses neural networks",
        "Natural language processing handles text",
        "Computer vision analyzes images",
    ]

    # Calculate embeddings in batch
    embeddings = calculate_embeddings_batch(
        texts=texts,
        model_name="all-MiniLM-L6-v2",
        backend="auto",
        force_cpu=True,
        show_progress=True,
    )

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Number of texts: {len(texts)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")


if __name__ == "__main__":
    # Run examples
    print("=== Single Text Example ===")
    example_single_text()

    print("\n=== Batch Texts Example ===")
    example_batch_texts()
