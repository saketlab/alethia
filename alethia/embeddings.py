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
    model: Optional[Union[str, Any]] = None,
    model_type: str = "sentence-transformer",
    client: Optional[Any] = None,
    api_key: Optional[str] = None,
    return_labels: bool = False,
    show_progress: bool = False,
    force_cpu: bool = True,
    cache_model: bool = True,
    **kwargs,
) -> Union[List[float], np.ndarray, Dict[str, List[float]]]:
    """
    Get embeddings for text(s) using various embedding models.
    Enhanced version with automatic model loading for sentence-transformers.

    Args:
        texts: Single text or list of texts to embed
        model: Model name (string) or loaded model object
        model_type: Type of model ("sentence-transformer", "fastembed", "openai", "google")
        client: Pre-configured client for API models
        api_key: API key for API models
        return_labels: Whether to return labels alongside embeddings
        show_progress: Show progress bar for batch processing
        force_cpu: Force CPU usage for local models
        cache_model: Cache loaded models for reuse
        **kwargs: Additional arguments for model loading

    Returns:
        Embeddings as numpy array, list, or dict depending on input and options
    """
    global _cached_models

    # Initialize model cache if it doesn't exist
    if "cache_model" in globals() and cache_model:
        if "_cached_models" not in globals():
            _cached_models = {}

    # Validate and prepare inputs
    if isinstance(texts, str):
        texts_list = [texts]
        is_single_input = True
    else:
        texts_list = list(texts)
        is_single_input = False

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
        import google.generativeai as genai

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
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "SentenceTransformers not available. Install with: pip install sentence-transformers"
            )

        # Handle both string model names and pre-loaded model objects
        if isinstance(model, str):
            # Check cache first
            cache_key = f"st_{model}_{force_cpu}"
            if (
                cache_model
                and "_cached_models" in globals()
                and cache_key in _cached_models
            ):
                model_obj = _cached_models[cache_key]
                if show_progress:
                    print(f"Using cached model: {model}")
            else:
                # Load the model
                model_obj = _load_sentence_transformer_model(
                    model_name=model,
                    force_cpu=force_cpu,
                    show_progress=show_progress,
                    **kwargs,
                )

                # Cache the model if requested
                if cache_model:
                    if "_cached_models" not in globals():
                        _cached_models = {}
                    _cached_models[cache_key] = model_obj

        elif isinstance(model, SentenceTransformer):
            model_obj = model
        elif model is None:
            raise ValueError(
                "Model must be provided as either a string name or SentenceTransformer object"
            )
        else:
            raise ValueError(
                "For sentence-transformer type, model must be a string name or SentenceTransformer object"
            )

        # Generate embeddings
        if show_progress and len(texts_list) > 1:
            embeddings = []
            for text in tqdm(texts_list, desc="Generating Embeddings"):
                embedding = model_obj.encode(text)
                embeddings.append(embedding)
            embeddings = np.stack(embeddings)
        else:
            # Use batch encoding for efficiency
            embeddings = model_obj.encode(
                texts_list, show_progress_bar=show_progress and len(texts_list) > 10
            )
            if len(texts_list) == 1:
                embeddings = embeddings.reshape(1, -1)
            else:
                embeddings = np.array(embeddings)

        if return_labels:
            labels_array = np.array(texts_list).reshape(-1, 1)
            embeddings = np.hstack([embeddings, labels_array])

        if is_single_input and not return_labels:
            return embeddings[0]
        else:
            return embeddings

    elif model_type.lower() == "fastembed":
        if not FASTEMBED_AVAILABLE:
            raise ImportError(
                "FastEmbed not available. Install with: pip install fastembed"
            )

        # Handle both string model names and pre-loaded model objects
        if isinstance(model, str):
            # Check cache first
            cache_key = f"fe_{model}"
            if (
                cache_model
                and "_cached_models" in globals()
                and cache_key in _cached_models
            ):
                model_obj = _cached_models[cache_key]
                if show_progress:
                    print(f"Using cached FastEmbed model: {model}")
            else:
                # Load the model
                model_obj = _load_fastembed_model(
                    model_name=model, show_progress=show_progress, **kwargs
                )

                # Cache the model if requested
                if cache_model:
                    if "_cached_models" not in globals():
                        _cached_models = {}
                    _cached_models[cache_key] = model_obj

        elif hasattr(model, "embed"):  # FastEmbed model object
            model_obj = model
        elif model is None:
            raise ValueError("FastEmbed model name or object must be provided")
        else:
            raise ValueError(
                "For fastembed type, model must be a string name or FastEmbed model object"
            )

        # Generate embeddings using FastEmbed
        if show_progress and len(texts_list) > 1:
            embeddings = []
            for text in tqdm(texts_list, desc="Generating FastEmbed Embeddings"):
                embedding = next(model_obj.embed([text]))
                embeddings.append(embedding)
            embeddings = np.stack(embeddings)
        else:
            embeddings = np.array(list(model_obj.embed(texts_list)))

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


def _load_sentence_transformer_model(
    model_name: str, force_cpu: bool = True, show_progress: bool = False, **kwargs
) -> SentenceTransformer:
    """
    Load a SentenceTransformer model with error handling

    Args:
        model_name: Name of the model to load
        force_cpu: Force CPU usage
        show_progress: Show loading progress
        **kwargs: Additional arguments for SentenceTransformer

    Returns:
        Loaded SentenceTransformer model
    """
    try:
        import torch

        # Determine device
        if force_cpu or not torch.cuda.is_available():
            device = "cpu"
            if show_progress:
                print(f"Loading {model_name} on CPU")
        else:
            device = "cuda"
            if show_progress:
                print(f"Loading {model_name} on GPU")

        # Load model with error handling
        try:
            model = SentenceTransformer(
                model_name, device=device, trust_remote_code=True, **kwargs
            )
            if show_progress:
                print(f"✅ Successfully loaded {model_name}")
            return model

        except RuntimeError as e:
            if "CUDA out of memory" in str(e) and not force_cpu:
                if show_progress:
                    print("⚠️ GPU memory error, trying CPU")
                model = SentenceTransformer(
                    model_name, device="cpu", trust_remote_code=True, **kwargs
                )
                if show_progress:
                    print(f"✅ Successfully loaded {model_name} on CPU")
                return model
            else:
                raise

    except Exception as e:
        logger.error(f"Failed to load SentenceTransformer model {model_name}: {e}")
        raise


def _load_fastembed_model(
    model_name: str, show_progress: bool = False, **kwargs
) -> Any:
    """
    Load a FastEmbed model with error handling

    Args:
        model_name: Name of the model to load
        show_progress: Show loading progress
        **kwargs: Additional arguments for FastEmbed

    Returns:
        Loaded FastEmbed model
    """
    try:
        from fastembed import TextEmbedding

        # Model name mapping for common aliases
        model_mapping = {
            "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
            "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
        }

        fastembed_model_name = model_mapping.get(model_name, model_name)

        if show_progress:
            print(f"Loading FastEmbed model: {fastembed_model_name}")

        model = TextEmbedding(model_name=fastembed_model_name, **kwargs)

        if show_progress:
            print(f"✅ Successfully loaded {fastembed_model_name} with FastEmbed")

        return model

    except Exception as e:
        logger.error(f"Failed to load FastEmbed model {model_name}: {e}")
        raise


def clear_model_cache():
    """Clear the cached models to free memory"""
    global _cached_models
    if "_cached_models" in globals():
        _cached_models.clear()
        print("✅ Model cache cleared")


def list_cached_models():
    """List currently cached models"""
    if "_cached_models" in globals():
        if _cached_models:
            print("Cached models:")
            for key in _cached_models.keys():
                print(f"  - {key}")
        else:
            print("No models currently cached")
    else:
        print("Model caching not initialized")


# Example usage functions with the enhanced get_embeddings
def example_usage():
    """Examples of how to use the enhanced get_embeddings function"""

    # Example 1: Simple usage with string model name
    print("Example 1: Simple usage with model name")
    labels = ["This is a test sentence", "Another example text", "Third sample"]

    embeddings = get_embeddings(
        texts=labels,
        model="sentence-transformers/all-MiniLM-L6-v2",
        model_type="sentence-transformer",
        return_labels=False,
        show_progress=True,
        cache_model=True,
    )
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"First embedding (first 5 values): {embeddings[0][:5]}")

    # Example 2: Single text
    print("\nExample 2: Single text embedding")
    single_embedding = get_embeddings(
        texts="Single text example",
        model="all-MiniLM-L6-v2",
        model_type="sentence-transformer",
        show_progress=True,
        cache_model=True,  # Will reuse cached model from Example 1
    )
    print(f"Single embedding shape: {single_embedding.shape}")

    # Example 3: With different model
    print("\nExample 3: Different model")
    try:
        embeddings2 = get_embeddings(
            texts=labels[:2],  # Just first 2 for speed
            model="all-mpnet-base-v2",
            model_type="sentence-transformer",
            show_progress=True,
            force_cpu=True,
        )
        print(f"Different model embeddings shape: {embeddings2.shape}")
    except Exception as e:
        print(f"Could not load alternative model: {e}")

    # Example 4: FastEmbed (if available)
    print("\nExample 4: FastEmbed")
    try:
        embeddings_fe = get_embeddings(
            texts=labels[:2],
            model="all-MiniLM-L6-v2",
            model_type="fastembed",
            show_progress=True,
        )
        print(f"FastEmbed embeddings shape: {embeddings_fe.shape}")
    except ImportError:
        print("FastEmbed not available")
    except Exception as e:
        print(f"FastEmbed error: {e}")

    # Show cached models
    print("\nCached models:")
    list_cached_models()


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

if __name__ == "__main__":
    example_usage()
