from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import psutil

# Gracefully handle optional imports
try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    # Create dummy classes/objects for type hints
    SentenceTransformer = None
    np = None
    tqdm = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False


def convert_memory_to_gb(memory_str):
    if pd.isna(memory_str) or memory_str == "Unknown":
        return None
    try:
        return round(float(memory_str) / 1024, 2)
    except (ValueError, TypeError):
        return None


def get_embeddings(
    texts: Union[str, List[str]],
    model_type: str = "sentence-transformer",
    model: Optional[Union[str, Any]] = None,  # Changed from SentenceTransformer to Any
    client: Optional[Any] = None,
    api_key: Optional[str] = None,
    return_labels: bool = False,
    show_progress: bool = False,
) -> Union[List[float], Any, Dict[str, List[float]]]:  # Changed np.ndarray to Any
    """
    Get embeddings for text(s) using various embedding models.

    Args:
        texts: Single text string or list of text strings to embed
        model_type: Type of model to use ('openai', 'google', or 'sentence-transformer')
        model: Model name (for OpenAI and Google) or SentenceTransformer object
        client: OpenAI client object (only for OpenAI)
        api_key: API key (only needed for OpenAI if client not provided)
        return_labels: Whether to return labels alongside embeddings (only for sentence-transformer)
        show_progress: Whether to display a progress bar (only for sentence-transformer)

    Returns:
        - For OpenAI: List of floats representing the embedding
        - For Google: A dictionary mapping each input text to its embedding
        - For SentenceTransformer: NumPy array of embeddings, with optional labels

    Raises:
        ValueError: If required parameters are missing or invalid
        ImportError: If required dependencies are not installed
    """
    # Validate and prepare inputs
    if isinstance(texts, str):
        texts_list = [texts]
    else:
        texts_list = texts

    is_single_input = isinstance(texts, str)

    # Generate embeddings based on model type
    if model_type.lower() == "openai":
        # OpenAI embeddings
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI library is not installed. Install it with: pip install openai"
            )

        if client is None and api_key is None:
            raise ValueError(
                "Either client or api_key must be provided for OpenAI embeddings"
            )

        if client is None:
            client = OpenAI(api_key=api_key)

        model_name = model or "text-embedding-ada-002"

        # For a single text, return the embedding directly
        if is_single_input:
            response = client.embeddings.create(input=texts_list, model=model_name)
            return response.data[0].embedding
        else:
            # For multiple texts, return a list of embeddings
            response = client.embeddings.create(input=texts_list, model=model_name)
            return [item.embedding for item in response.data]

    elif model_type.lower() == "google":
        # Google's Generative AI embeddings
        try:
            import genai  # Google's generative AI library
        except ImportError:
            raise ImportError(
                "Google GenerativeAI library is not installed. Install it with: pip install google-generativeai"
            )

        if model is None:
            raise ValueError("Model name must be provided for Google embeddings")

        # For a single text, return the embedding directly
        if is_single_input:
            return genai.embed_content(model=model, content=texts)["embedding"]
        else:
            # For multiple texts, return a dictionary mapping each text to its embedding
            return {
                text: genai.embed_content(model=model, content=text)["embedding"]
                for text in texts_list
            }

    elif model_type.lower() == "sentence-transformer":
        # Check if SentenceTransformers is available
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "SentenceTransformers library is not installed. Install it with: pip install sentence-transformers"
            )

        if model is None:
            raise ValueError("SentenceTransformer model object must be provided")

        if not isinstance(model, SentenceTransformer):
            raise ValueError(
                "For sentence-transformer type, model must be a SentenceTransformer object"
            )

        # Generate embeddings with or without progress bar
        if show_progress:
            embeddings = np.stack(
                [
                    model.encode(text)
                    for text in tqdm(texts_list, desc="Generating Embeddings")
                ]
            )
        else:
            embeddings = np.stack([model.encode(text) for text in texts_list])

        # Add labels if requested
        if return_labels:
            labels_array = np.array(texts_list).reshape(-1, 1)
            embeddings = np.hstack([embeddings, labels_array])

        # For a single text, return the embedding directly (but keep as numpy array)
        if is_single_input and not return_labels:
            return embeddings[0]
        else:
            return embeddings

    else:
        raise ValueError(
            "model_type must be one of: 'openai', 'google', or 'sentence-transformer'"
        )


def setup_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "Matplotlib is not installed. Install it with: pip install matplotlib"
        )

    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial"]
    plt.rcParams["axes.labelweight"] = "normal"

    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = "Arial"
    plt.rcParams["mathtext.it"] = "Arial:italic"
    plt.rcParams["mathtext.bf"] = "Arial:bold"


def get_system_usage() -> Dict[str, float]:
    """
    Get current system resource usage.

    Returns:
        Dict[str, float]: Dictionary containing CPU and RAM usage percentages.
    """
    usage = {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "ram_percent": psutil.virtual_memory().percent,
    }

    return usage


def get_gpu_usage() -> Dict[str, Union[str, float]]:
    """
    Get current GPU usage if available.

    Returns:
        Dict[str, Union[str, float]]: Dictionary containing GPU information or None if no GPU.
    """
    try:
        import torch
    except ImportError:
        return {"available": False, "error": "PyTorch not installed"}

    if not torch.cuda.is_available():
        return {"available": False}

    gpu_device = torch.cuda.current_device()
    gpu_info = {
        "available": True,
        "device": gpu_device,
        "name": torch.cuda.get_device_name(gpu_device),
        "memory_used_gb": torch.cuda.memory_allocated(gpu_device) / (1024**3),
        "memory_total_gb": torch.cuda.get_device_properties(gpu_device).total_memory
        / (1024**3),
    }

    return gpu_info


def print_resource_usage(prefix: str = ""):
    """
    Print current system and GPU resource usage.

    Args:
        prefix (str): Optional prefix string for the output (e.g., "Initial" or "Final").
    """
    label = f"{prefix} " if prefix else ""

    sys_usage = get_system_usage()
    print(f"{label}CPU Usage: {sys_usage['cpu_percent']}%")
    print(f"{label}RAM Usage: {sys_usage['ram_percent']}%")

    gpu_info = get_gpu_usage()
    if gpu_info.get("available"):
        print(f"GPU: {gpu_info['name']}")
        print(
            f"{label}GPU Memory Usage: {gpu_info['memory_used_gb']:.2f}GB / {gpu_info['memory_total_gb']:.2f}GB"
        )
    else:
        if not prefix:
            if "error" in gpu_info:
                print(f"GPU check failed: {gpu_info['error']}")
            else:
                print("No GPU detected, running on CPU only")

    return sys_usage, gpu_info
