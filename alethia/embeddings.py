import logging
from typing import Any, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

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

_cached_models = {}


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
) -> Union[List[float], np.ndarray]:
    """Get embeddings for text(s) using various embedding models."""
    texts_list = [texts] if isinstance(texts, str) else list(texts)
    is_single = isinstance(texts, str)

    if model_type.lower() == "openai":
        from openai import OpenAI

        if client is None and api_key is None:
            raise ValueError("Either client or api_key required for OpenAI")
        if client is None:
            client = OpenAI(api_key=api_key)
        model_name = model or "text-embedding-ada-002"
        response = client.embeddings.create(input=texts_list, model=model_name)
        if is_single:
            return response.data[0].embedding
        return [item.embedding for item in response.data]

    elif model_type.lower() == "google":
        import google.generativeai as genai

        if model is None:
            raise ValueError("Model name required for Google embeddings")
        if is_single:
            return genai.embed_content(model=model, content=texts)["embedding"]
        return {
            text: genai.embed_content(model=model, content=text)["embedding"]
            for text in texts_list
        }

    elif model_type.lower() == "sentence-transformer":
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("SentenceTransformers not available")

        if isinstance(model, str):
            cache_key = f"st_{model}_{force_cpu}"
            if cache_model and cache_key in _cached_models:
                model_obj = _cached_models[cache_key]
            else:
                model_obj = _load_sentence_transformer(model, force_cpu)
                if cache_model:
                    _cached_models[cache_key] = model_obj
        elif isinstance(model, SentenceTransformer):
            model_obj = model
        else:
            raise ValueError("Model must be string or SentenceTransformer")

        embeddings = model_obj.encode(
            texts_list, show_progress_bar=show_progress and len(texts_list) > 10
        )
        if len(texts_list) == 1:
            embeddings = embeddings.reshape(1, -1)

        if return_labels:
            labels = np.array(texts_list).reshape(-1, 1)
            embeddings = np.hstack([embeddings, labels])

        return embeddings[0] if is_single and not return_labels else embeddings

    elif model_type.lower() == "fastembed":
        if not FASTEMBED_AVAILABLE:
            raise ImportError("FastEmbed not available")

        if isinstance(model, str):
            cache_key = f"fe_{model}"
            if cache_model and cache_key in _cached_models:
                model_obj = _cached_models[cache_key]
            else:
                model_obj = _load_fastembed(model)
                if cache_model:
                    _cached_models[cache_key] = model_obj
        elif hasattr(model, "embed"):
            model_obj = model
        else:
            raise ValueError("Model must be string or FastEmbed model")

        embeddings = np.array(list(model_obj.embed(texts_list)))

        if return_labels:
            labels = np.array(texts_list).reshape(-1, 1)
            embeddings = np.hstack([embeddings, labels])

        return embeddings[0] if is_single and not return_labels else embeddings

    raise ValueError(
        "model_type must be: 'openai', 'google', 'sentence-transformer', or 'fastembed'"
    )


def _load_sentence_transformer(model_name: str, force_cpu: bool = True):
    import torch

    device = "cpu" if force_cpu or not torch.cuda.is_available() else "cuda"
    try:
        return SentenceTransformer(model_name, device=device, trust_remote_code=True)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e) and not force_cpu:
            return SentenceTransformer(model_name, device="cpu", trust_remote_code=True)
        raise


def _load_fastembed(model_name: str):
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=model_name)


def clear_model_cache():
    _cached_models.clear()
