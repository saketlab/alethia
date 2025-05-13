import numpy as np
import pandas as pd
from typing import Union, List
from tqdm import tqdm
from .utils import print_resource_usage
from sentence_transformers import SentenceTransformer
import time
import requests
import sys
import torch
import numpy as np
from tqdm import tqdm
from typing import Union, List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer


def load_sentence_transformer(model_name, force_cpu=False, **kwargs):
    """
    Load a sentence transformer model with error handling for CUDA out of memory errors.

    Args:
        model_name (str): Name of the model to load
        force_cpu (bool): If True, forces model to load on CPU even if GPU is available

    Returns:
        SentenceTransformer model or None if loading failed
    """
    try:
        if force_cpu:
            print(f"Loading model {model_name} on CPU (forced)")
            device = torch.device("cpu")
            st_model = SentenceTransformer(
                model_name, device=device, trust_remote_code=True, **kwargs
            )
        else:
            print(f"Loading model {model_name}")
            st_model = SentenceTransformer(model_name, trust_remote_code=True, **kwargs)

        print(f"Successfully loaded model: {model_name}")
        return st_model

    except RuntimeError as e:
        error_str = str(e)
        if "CUDA out of memory" in error_str or "out of memory" in error_str:
            print(f"GPU memory error: {error_str}", file=sys.stderr)
            print("Trying to load model on CPU instead...", file=sys.stderr)

            try:
                # Attempt to load on CPU
                device = torch.device("cpu")
                st_model = SentenceTransformer(
                    model_name, device=device, trust_remote_code=True, **kwargs
                )
                print(f"Successfully loaded model {model_name} on CPU", file=sys.stderr)
                return st_model
            except Exception as cpu_e:
                print(f"Failed to load model on CPU: {str(cpu_e)}", file=sys.stderr)
                print("Memory optimization tips:", file=sys.stderr)
                print(
                    "1. Free up GPU memory by closing other applications",
                    file=sys.stderr,
                )
                print("2. Try a smaller model like 'all-MiniLM-L6-v2'", file=sys.stderr)
                print(
                    "3. Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True environment variable",
                    file=sys.stderr,
                )
                return None

        else:
            print(f"Error loading model: {error_str}", file=sys.stderr)
            return None

    except Exception as e:
        print(f"Error loading model: {str(e)}", file=sys.stderr)
        print(
            "Please check the model name or try one of the recommended models:",
            file=sys.stderr,
        )
        print("- 'all-MiniLM-L6-v2'", file=sys.stderr)
        print("- 'all-mpnet-base-v2'", file=sys.stderr)
        print("- 'paraphrase-multilingual-MiniLM-L12-v2'", file=sys.stderr)
        return None


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate the cosine similarity between two vectors.

    Parameters
    ----------
    vec1 : np.ndarray
        First input vector.
    vec2 : np.ndarray
        Second input vector.

    Returns
    -------
    float
        Cosine similarity between the two vectors.
    """

    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def run_st(
    model_input: Union[str, SentenceTransformer],
    incorrect_entries: List[str],
    reference_entries: List[str],
    force_cpu: bool = False,
) -> pd.DataFrame:
    """
    Run the ST model with either a model name or initialized SentenceTransformer object.

    Args:
        model_input (Union[str, SentenceTransformer]): Either a model name string or an initialized SentenceTransformer object.
        incorrect_entries (List[str]): A list of incorrect entries.
        reference_entries (List[str]): A list of reference entries.
        force_cpu (bool): If True, forces model to load on CPU even if GPU is available

    Returns:
        pd.DataFrame: A DataFrame containing the results.
    """
    # Print initial resource usage
    print("Initial resource usage:")
    print_resource_usage()

    # Start timing
    start_time = time.time()

    # Handle either string model name or initialized SentenceTransformer object
    if isinstance(model_input, str):
        st_model = load_sentence_transformer(model_input, force_cpu)
        if st_model is None:
            return pd.DataFrame(
                columns=["given_entity", "alethia_prediction", "alethia_score"]
            )
    else:
        # Assume it's already an initialized SentenceTransformer model
        st_model = model_input


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
        from openai import OpenAI

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
        import genai  # Google's generative AI library

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
        # SentenceTransformer embeddings
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


def run_st(
    incorrect_entries: List[str],
    reference_entries: List[str],
    model: Union[str, SentenceTransformer] = "Salesforce/SFR-Embedding-Mistral",
    force_cpu: bool = False,
) -> pd.DataFrame:
    """
    Run the ST model with either a model name or initialized SentenceTransformer object.

    Args:
        model (Union[str, SentenceTransformer]): Either a model name string or an initialized SentenceTransformer object.
        incorrect_entries (List[str]): A list of incorrect entries.
        reference_entries (List[str]): A list of reference entries.

    Returns:
        pd.DataFrame: A DataFrame containing the results.
    """
    if isinstance(model, str):
        try:
            st_model = load_sentence_transformer(model_name=model)
        except Exception as e:
            print(f"Error loading model: {str(e)}", file=sys.stderr)
            print(
                "Please check the model name or try one of the recommended models.",
                file=sys.stderr,
            )
            return pd.DataFrame(
                columns=["given_entity", "alethia_prediction", "alethia_score"]
            )
    else:
        st_model = model
    print("Initial resource usage:")
    print_resource_usage()

    start_time = time.time()
    print(f"Computing reference embeddings for {len(reference_entries)} entries...")
    reference_embeddings = {
        ref_entity: st_model.encode(ref_entity) for ref_entity in reference_entries
    }
    print(f"Processing {len(incorrect_entries)} incorrect entries...")
    results = []
    for incorrect in tqdm(incorrect_entries, desc="Processing entries", unit="entry"):
        if str(incorrect) == "nan":
            results.append((incorrect, np.nan, np.nan))
            continue
        query_embedding = st_model.encode(incorrect)

        similarities = {
            state: cosine_similarity(query_embedding, emb)
            for state, emb in reference_embeddings.items()
        }

        best_match = max(similarities, key=similarities.get)
        best_score = similarities[best_match]

        results.append((incorrect, best_match, best_score))

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Processing completed in {elapsed_time:.2f} seconds")
    results_df = pd.DataFrame(
        results,
        columns=[
            "given_entity",
            "alethia_prediction",
            "alethia_score",
        ],
    )
    return results_df


def run_rapidfuzz(incorrect_entries: list, reference_entries: list) -> pd.DataFrame:
    """
    Run the RapidFuzz model.
    Args:
        incorrect_entries (list): A list of incorrect entries.
        reference_entries (list): A list of reference entries.
    Returns:
        pd.DataFrame: A DataFrame containing the results.
    """

    from rapidfuzz import fuzz, process

    results = []
    for incorrect in incorrect_entries:
        # Get best match using token sort ratio
        best_match, score, _ = process.extractOne(
            incorrect, reference_entries, scorer=fuzz.token_sort_ratio
        )
        results.append((incorrect, best_match, score / 100))

    return pd.DataFrame(
        results, columns=["given_entity", "alethia_prediction", "alethia_score"]
    )


def run_gemini(
    incorrect_entries: list,
    reference_entries: list,
    model_name: str = "models/embedding-001",
) -> pd.DataFrame:
    """
    Run the Gemini model.
    Args:
        incorrect_entries (list): A list of incorrect entries.
        reference_entries (list): A list of reference entries.
        model_name (str): The name of the model to use. Default is "models/embedding-001".
    Returns:
        pd.DataFrame: A DataFrame containing the results.
    """
    # Configure Gemini API
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key is None:
        raise ValueError("GEMINI_API_KEY not set in environment variables.")
    genai.configure(api_key=api_key)

    reference_embeddings = {
        state: genai.embed_content(model=model_name, content=state)["embedding"]
        for state in reference_entries
    }

    results = []

    for incorrect in incorrect_entries:
        query_embedding = genai.embed_content(model=model_name, content=incorrect)[
            "embedding"
        ]

        similarities = {
            state: cosine_similarity(query_embedding, emb)
            for state, emb in reference_embeddings.items()
        }

        best_match = max(similarities, key=similarities.get)
        best_score = similarities[best_match]

        results.append((incorrect, best_match, best_score))

    results_df = pd.DataFrame(
        results, columns=["given_entity", "alethia_prediction", "alethia_score"]
    )

    return results_df


def run_openai(incorrect_entries: list, reference_entries: list) -> pd.DataFrame:
    """
    Run the OpenAI model.

    Args:
        incorrect_entries (list): A list of incorrect entries.
        reference_entries (list): A list of reference entries.

    Returns:
        pd.DataFrame: A DataFrame containing the results.
    """
    # Read API key from environment
    api_key = os.getenv("OPENAI_API_KEY")

    if api_key is None:
        raise ValueError("OPENAI_API_KEY not set in environment variables.")

    client = OpenAI(api_key=api_key)

    model_name = "text-embedding-ada-002"  # Embedding model

    # Get embeddings for reference entries
    reference_embeddings = {
        state: get_embedding(client, state, model=model_name)
        for state in reference_entries
    }

    results = []

    for incorrect in incorrect_entries:
        query_embedding = get_embedding(client, incorrect, model=model_name)

        # Compute similarities
        similarities = {
            state: cosine_similarity(query_embedding, emb)
            for state, emb in reference_embeddings.items()
        }

        best_match = max(similarities, key=similarities.get)
        best_score = similarities[best_match]

        results.append((incorrect, best_match, best_score))

    results_df = pd.DataFrame(
        results, columns=["given_entity", "alethia_prediction", "alethia_score"]
    )

    return results_df


def alethia(
    incorrect_entries: list,
    reference_entries: list,
    model: str = "Salesforce/SFR-Embedding-Mistral",
    verbose=True,
    **kwargs,
) -> pd.DataFrame:
    """Run the Alethia model.
    Args:
        incorrect_entries (list): A list of incorrect entries.
        reference_entries (list): A list of reference entries.
        model (str, optional): _description_. Defaults to "Salesforce/SFR-Embedding-Mistral".

    Raises:
        ValueError: _description_

    Returns:
        pd.DataFrame: _description_
    """
    if model == "gemini":
        # pass arguments strictly
        return run_gemini(
            incorrect_entries=incorrect_entries,
            reference_entries=reference_entries,
            **kwargs,
        )
    elif model == "openai":
        return run_openai(
            incorrect_entries=incorrect_entries,
            reference_entries=reference_entries,
            **kwargs,
        )
    elif model == "rapidfuzz":
        return run_rapidfuzz(incorrect_entries, reference_entries)
    else:
        if verbose:
            print(f"Initializing model {model}")
        st_model = load_sentence_transformer(model_name=model)
        if st_model is not None:
            return run_st(
                incorrect_entries=incorrect_entries,
                reference_entries=reference_entries,
                model=st_model,
            )
        else:
            raise ValueError(
                f"Model {model} not found in Hugging Face or not supported."
            )


def benchmark_models(models_list: list, reference_values: list):
    embeddings = []
    for model in models_list:
        get_embedding
