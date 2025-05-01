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

def load_sentence_transformer(model_name, force_cpu=False):
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
                model_name, device=device, trust_remote_code=True
            )
        else:
            print(f"Loading model {model_name}")
            st_model = SentenceTransformer(model_name, trust_remote_code=True)

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
                    model_name, device=device, trust_remote_code=True
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


def get_embedding(
    input_list: list,
    st_model: SentenceTransformer,
    return_labels=False,
    show_progress=False,
) -> np.ndarray:
    """
    Generate embeddings for a list of input strings using a SentenceTransformer model.

    Args:
        input_list (list): List of strings to embed.
        st_model (SentenceTransformer): The SentenceTransformer model to use.
        return_labels (bool, optional): Whether to return labels alongside embeddings. Defaults to False.
        show_progress_bar (bool, optional): Whether to display a progress bar. Defaults to False.

    Returns:
        np.ndarray: A NumPy array of embeddings, where each row corresponds to an input string.
                      If return_labels is True, the last column will contain the labels.
    """
    if show_progress:
        embeddings = np.stack(
            [
                st_model.encode(entity)
                for entity in tqdm(input_list, desc="Generating Embeddings")
            ]
        )
    else:
        embeddings = np.stack([st_model.encode(entity) for entity in input_list])

    if return_labels:
        labels_array = np.array(input_list).reshape(-1, 1)
        embeddings = np.hstack([embeddings, labels_array])
    return embeddings


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
        # Get embedding for the query
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


# Write numpy style documentation for the method below
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
