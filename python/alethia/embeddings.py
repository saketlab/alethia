import importlib.util
import logging
from typing import Any

import numpy as np
from tqdm import tqdm

SENTENCE_TRANSFORMERS_AVAILABLE = (
    importlib.util.find_spec("sentence_transformers") is not None
)

try:
    import fastembed  # noqa: F401

    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False

logger = logging.getLogger(__name__)


def get_embeddings(
    texts: str | list[str],
    model: str | Any | None = None,
    model_type: str = "sentence-transformer",
    client: Any | None = None,
    api_key: str | None = None,
    return_labels: bool = False,
    show_progress: bool = False,
    force_cpu: bool = True,
    cache_model: bool = True,
    **kwargs,
) -> list[float] | np.ndarray | dict[str, list[float]]:
    """Embed text, loading the model by name when one is not already given."""
    global _cached_models

    if "cache_model" in globals() and cache_model:
        if "_cached_models" not in globals():
            _cached_models = {}

    if isinstance(texts, str):
        texts_list = [texts]
        is_single_input = True
    else:
        texts_list = list(texts)
        is_single_input = False

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

        if isinstance(model, str):
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
                model_obj = _load_sentence_transformer_model(
                    model_name=model,
                    force_cpu=force_cpu,
                    show_progress=show_progress,
                    **kwargs,
                )

                if cache_model:
                    if "_cached_models" not in globals():
                        _cached_models = {}
                    _cached_models[cache_key] = model_obj

        elif hasattr(model, "encode"):
            model_obj = model
        elif model is None:
            raise ValueError(
                "Model must be provided as either a string name or SentenceTransformer object"
            )
        else:
            raise ValueError(
                "For sentence-transformer type, model must be a string name or SentenceTransformer object"
            )

        if show_progress and len(texts_list) > 1:
            embeddings = []
            for text in tqdm(texts_list, desc="Generating Embeddings"):
                embedding = model_obj.encode(text)
                embeddings.append(embedding)
            embeddings = np.stack(embeddings)
        else:
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

        if isinstance(model, str):
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
                model_obj = _load_fastembed_model(
                    model_name=model, show_progress=show_progress, **kwargs
                )

                if cache_model:
                    if "_cached_models" not in globals():
                        _cached_models = {}
                    _cached_models[cache_key] = model_obj

        elif hasattr(model, "embed"):
            model_obj = model
        elif model is None:
            raise ValueError("FastEmbed model name or object must be provided")
        else:
            raise ValueError(
                "For fastembed type, model must be a string name or FastEmbed model object"
            )

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
) -> Any:
    """Load a SentenceTransformer by name, raising a readable error on failure."""
    from sentence_transformers import SentenceTransformer

    try:
        import torch

        if force_cpu or not torch.cuda.is_available():
            device = "cpu"
            if show_progress:
                print(f"Loading {model_name} on CPU")
        else:
            device = "cuda"
            if show_progress:
                print(f"Loading {model_name} on GPU")

        try:
            model = SentenceTransformer(
                model_name, device=device, trust_remote_code=True, **kwargs
            )
            if show_progress:
                print(f"[ok] Successfully loaded {model_name}")
            return model

        except RuntimeError as e:
            if "CUDA out of memory" in str(e) and not force_cpu:
                if show_progress:
                    print("[!] GPU memory error, trying CPU")
                model = SentenceTransformer(
                    model_name, device="cpu", trust_remote_code=True, **kwargs
                )
                if show_progress:
                    print(f"[ok] Successfully loaded {model_name} on CPU")
                return model
            else:
                raise

    except Exception as e:
        logger.error(f"Failed to load SentenceTransformer model {model_name}: {e}")
        raise


def _load_fastembed_model(
    model_name: str, show_progress: bool = False, **kwargs
) -> Any:
    """Load a FastEmbed model by name, raising a readable error on failure."""
    try:
        from fastembed import TextEmbedding

        model_mapping = {
            "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
            "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
        }

        fastembed_model_name = model_mapping.get(model_name, model_name)

        if show_progress:
            print(f"Loading FastEmbed model: {fastembed_model_name}")

        model = TextEmbedding(model_name=fastembed_model_name, **kwargs)

        if show_progress:
            print(f"[ok] Successfully loaded {fastembed_model_name} with FastEmbed")

        return model

    except Exception as e:
        logger.error(f"Failed to load FastEmbed model {model_name}: {e}")
        raise
