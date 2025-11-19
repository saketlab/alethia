"""
Clinical NLP functions using LLMs for medical text processing.

This module provides functions for clinical text analysis including ICD-10 code
mapping using instruction-following clinical LLMs.

Supports multiple model backends:
- Ollama (local models via instructor)
- HuggingFace (local models via transformers)
- OpenAI (cloud API via instructor)
"""

import importlib.resources
import json
import logging
import re
from difflib import get_close_matches
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from pydantic import BaseModel, Field, field_validator

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.debug("transformers not available")

try:
    import instructor

    INSTRUCTOR_AVAILABLE = True
except ImportError:
    INSTRUCTOR_AVAILABLE = False
    logger.debug("instructor not available")


SEMANTIC_MODEL_NAME = "FremyCompany/BioLORD-2023"
_SEMANTIC_MODEL = None
_SEMANTIC_MODEL_UNAVAILABLE = False


def _get_semantic_model():
    """Load and cache the sentence-transformers model used for semantic search."""

    global _SEMANTIC_MODEL, _SEMANTIC_MODEL_UNAVAILABLE

    if _SEMANTIC_MODEL_UNAVAILABLE:
        return None

    if _SEMANTIC_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer

            logger.debug("Loading semantic similarity model: %s", SEMANTIC_MODEL_NAME)
            _SEMANTIC_MODEL = SentenceTransformer(SEMANTIC_MODEL_NAME)
        except ImportError:
            logger.warning(
                "sentence-transformers not available; install with `pip install sentence-transformers` for improved ICD-10 matching"
            )
            _SEMANTIC_MODEL_UNAVAILABLE = True
            return None
        except Exception as exc:
            logger.warning(
                "Failed to load semantic similarity model (%s): %s", SEMANTIC_MODEL_NAME, exc
            )
            _SEMANTIC_MODEL_UNAVAILABLE = True
            return None

    return _SEMANTIC_MODEL


def _normalize_cosine_similarity(score: float) -> float:
    """Normalize cosine similarity (-1 to 1) into [0, 1] range."""

    return max(0.0, min(1.0, (score + 1.0) / 2.0))


def _semantic_similarity_ranking(
    clinical_text: str, candidates_df: pd.DataFrame
) -> pd.DataFrame:
    """Rank candidate ICD-10 rows by semantic similarity to the clinical text."""

    if not clinical_text or candidates_df.empty:
        return pd.DataFrame()

    model = _get_semantic_model()
    if model is None:
        return pd.DataFrame()

    try:
        from sentence_transformers import util
        import torch
    except ImportError:
        return pd.DataFrame()

    candidate_subset = (
        candidates_df.dropna(subset=["description2"])
        .copy()
        .astype({"description2": str})
    )
    candidate_subset["description2"] = candidate_subset["description2"].str.strip()
    candidate_subset = candidate_subset[candidate_subset["description2"] != ""]
    candidate_subset = candidate_subset.drop_duplicates(subset=["description2"]).reset_index(
        drop=True
    )

    if candidate_subset.empty:
        return pd.DataFrame()

    try:
        query_embedding = model.encode(clinical_text, convert_to_tensor=True)
        description_embeddings = model.encode(
            candidate_subset["description2"].tolist(),
            convert_to_tensor=True,
            show_progress_bar=False,
        )

        similarities = util.cos_sim(query_embedding, description_embeddings)[0]
        similarity_values = similarities.detach().cpu().numpy()
        candidate_subset["similarity"] = similarity_values
        candidate_subset.sort_values(
            by="similarity", ascending=False, inplace=True, ignore_index=True
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return candidate_subset
    except Exception as exc:
        logger.debug("Semantic similarity ranking failed: %s", exc)
        return pd.DataFrame()


# Pydantic models for structured ICD-10 outputs
class ICD10Match(BaseModel):
    """Response model for ICD-10 category matching based on description2."""

    matched_description2: str = Field(
        description="The matched disease/condition from ICD-10 description2 column"
    )
    confidence: float = Field(
        description="Confidence score between 0 and 1", ge=0.0, le=1.0
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Brief explanation of why this category was selected",
    )

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, value: Any) -> float:
        """Normalize confidence into the [0.0, 1.0] range."""
        if value is None:
            logger.warning("Confidence value missing; defaulting to 0.0")
            return 0.0

        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("Confidence must be a numeric value") from exc

        if numeric < 0.0 or numeric > 1.0:
            logger.warning(
                "Received out-of-range confidence %.2f; clamping to [0, 1]", numeric
            )

        return max(0.0, min(numeric, 1.0))


class HuggingFaceLLMClient:
    """
    Wrapper for HuggingFace transformers models to provide instructor-like interface.

    This allows using local HuggingFace models (like MedGemma) with the same
    interface as instructor-based clients.
    """

    def generate_structured_output(
        self, messages: List[Dict[str, str]], response_model: type[BaseModel]
    ) -> BaseModel:
        """
        Generate structured output from messages.

        Args:
            messages: List of message dicts with 'role' and 'content'
            response_model: Pydantic model class for structured output

        Returns:
            Instance of response_model with parsed output
        """
        # Use outlines for constrained JSON generation
        try:
            import outlines
        except ImportError:
            raise ImportError(
                "outlines library is required for HuggingFace structured output. "
                "Install with: pip install outlines"
            )

        # Build prompt from messages
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt_parts.append(
                    f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>"
                )
            elif role == "user":
                prompt_parts.append(
                    f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
                )

        prompt_parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        prompt = "".join(prompt_parts)

        # Use outlines for constrained generation
        logger.info("Using outlines for constrained JSON generation...")

        # Create model wrapper for outlines
        from outlines import models

        model_wrapper = models.Transformers(self.model, self.tokenizer)
        generator = outlines.generate.json(model_wrapper, response_model)

        # Generate with constrained decoding
        try:
            result = generator(prompt, max_tokens=512)
            return result
        except Exception as e:
            logger.error(f"Constrained generation failed: {e}")
            raise ValueError(f"Could not generate structured output: {e}")

    class ChatCompletions:
        """Inner class to mimic instructor's chat.completions interface."""

        def __init__(self, parent):
            self.parent = parent

        def create(
            self,
            model: str,
            response_model: type[BaseModel],
            messages: List[Dict[str, str]],
        ) -> BaseModel:
            """Create completion with structured output."""
            return self.parent.generate_structured_output(messages, response_model)

    class Chat:
        """Inner class to mimic instructor's chat interface."""

        def __init__(self, parent):
            self.completions = HuggingFaceLLMClient.ChatCompletions(parent)

    def __init__(self, model_name: str, device: Optional[str] = None):
        """Initialize with model loading."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library is not installed. "
                "Install it with: pip install transformers torch"
            )

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading HuggingFace model: {model_name} on {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )

        if self.device == "cpu":
            self.model = self.model.to(self.device)

        # Initialize chat interface
        self.chat = HuggingFaceLLMClient.Chat(self)

        logger.info(f"Model loaded successfully on {self.device}")


def load_icd10_data(limit: Optional[int] = None) -> pd.DataFrame:
    """
    Load ICD-10 WHO classification data from package resources.

    Args:
        limit: Optional limit on number of records to load (for testing)

    Returns:
        DataFrame with ICD-10 codes and descriptions

    Raises:
        FileNotFoundError: If the ICD-10 data file is not found
    """
    try:
        # Use context manager for package resources
        with importlib.resources.path("alethia.data", "icd10_who.csv") as data_path:
            if limit:
                df = pd.read_csv(data_path, nrows=limit)
            else:
                df = pd.read_csv(data_path)

            logger.info(f"Loaded {len(df)} ICD-10 codes from WHO classification")
            return df

    except FileNotFoundError:
        logger.error("ICD-10 data file not found in package resources")
        raise


def get_clinical_llm_client(
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    device: Optional[str] = None,
) -> Any:
    """
    Get client configured for clinical LLM (Ollama, HuggingFace, or OpenAI).

    Supports multiple model backends:
    - Ollama: Local models via instructor (prefix: 'ollama/')
    - HuggingFace: Local models via transformers (prefix: 'huggingface/' or 'hf/')
    - OpenAI: Cloud API (prefix: 'openai/')

    Args:
        model_name: Model identifier. Defaults to 'huggingface/aaditya/Llama3-OpenBioLLM-8B'
                   Examples:
                   - 'huggingface/aaditya/Llama3-OpenBioLLM-8B' (default, HuggingFace local)
                   - 'ollama/koesn/llama3-openbiollm-8b' (Ollama local)
                   - 'ollama/medllama2' (Ollama, official library)
                   - 'huggingface/google/medgemma-4b-it' (HuggingFace local)
                   - 'hf/google/medgemma-27b-it' (HuggingFace local, shorthand)
                   - 'openai/gpt-4' (OpenAI API, requires API key)
        api_key: API key if using cloud provider (OpenAI, etc.)
        device: Device for HuggingFace models ('cuda', 'cpu', or None for auto)

    Returns:
        Client instance (instructor client or HuggingFaceLLMClient)

    Raises:
        ImportError: If required libraries are not installed
    """
    # Default to OpenBioLLM-8B via Ollama (most reliable)
    if model_name is None:
        model_name = "ollama/koesn/llama3-openbiollm-8b"
        logger.info(
            f"Using default clinical model: {model_name}. "
            "Ensure Ollama is running with: ollama pull koesn/llama3-openbiollm-8b"
        )

    # Detect model backend
    if model_name.startswith("huggingface/") or model_name.startswith("hf/"):
        # HuggingFace model
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers and torch libraries are required for HuggingFace models. "
                "Install with: pip install transformers torch"
            )

        # Extract model name (remove prefix)
        hf_model_name = model_name.replace("huggingface/", "").replace("hf/", "")
        logger.info(f"Using HuggingFace model: {hf_model_name}")

        return HuggingFaceLLMClient(model_name=hf_model_name, device=device)

    else:
        # Ollama or OpenAI (use instructor)
        if not INSTRUCTOR_AVAILABLE:
            raise ImportError(
                "instructor library is required for Ollama/OpenAI models. "
                "Install with: pip install instructor"
            )

        try:
            client = instructor.from_provider(model_name, api_key=api_key)
            logger.info(f"Using instructor-based model: {model_name}")
            return client
        except Exception as e:
            logger.error(f"Failed to create clinical LLM client: {e}")
            raise


def build_icd10_context(
    icd10_df: pd.DataFrame,
    category_filter: Optional[str] = None,
    max_codes: int = 500,
    clinical_text: Optional[str] = None,
    use_semantic_search: bool = True,
) -> tuple[str, pd.DataFrame]:
    """
    Build context string from ICD-10 codes for LLM prompt using two-stage filtering.

    Stage 1: Filter description3 categories using semantic search
    Stage 2: Get all description2 values from those categories and present to LLM

    If clinical_text is provided and use_semantic_search=True, uses semantic similarity
    to find the most relevant description3 categories, then returns all corresponding
    description2 values for fine-grained matching.

    Args:
        icd10_df: DataFrame with ICD-10 codes
        category_filter: Optional category prefix to filter (e.g., 'A0' for certain infectious diseases)
        max_codes: Maximum number of description3 categories to include in stage 1 filtering
        clinical_text: Optional clinical text for semantic search
        use_semantic_search: Whether to use semantic search (requires clinical_text)

    Returns:
        Tuple of (formatted string with description2 values, filtered DataFrame for lookup)
    """
    if category_filter:
        filtered_df = icd10_df[icd10_df["icd_fullcode"].str.startswith(category_filter)]
    else:
        filtered_df = icd10_df

    # STAGE 1: Filter by description3 to get relevant categories
    # Group by description3 to get unique categories
    unique_desc3 = filtered_df.drop_duplicates(subset=["description3"])

    # If we have clinical text and semantic search is enabled, use it to find relevant categories
    if clinical_text and use_semantic_search and len(unique_desc3) > max_codes:
        try:
            from sentence_transformers import util
            import torch

            logger.info(
                f"Stage 1: Using semantic search on description3 to find top {max_codes} most relevant ICD-10 categories..."
            )

            model = _get_semantic_model()
            if model is None:
                raise RuntimeError("Semantic model unavailable")

            categories = unique_desc3["description3"].tolist()
            query_embedding = model.encode(clinical_text, convert_to_tensor=True)
            category_embeddings = model.encode(
                categories, convert_to_tensor=True, show_progress_bar=False
            )

            similarities = util.cos_sim(query_embedding, category_embeddings)[0]

            top_k = min(max_codes, len(categories))
            top_k_indices = (
                torch.topk(similarities, k=top_k).indices.cpu().numpy()
            )

            unique_desc3 = unique_desc3.iloc[top_k_indices].copy()

            logger.info(
                f"Stage 1: Selected top {len(unique_desc3)} semantically relevant description3 categories"
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except ImportError:
            logger.warning(
                "sentence-transformers not available for semantic search. "
                "Install with: pip install sentence-transformers"
            )
            # Fall back to random sampling
            if len(unique_desc3) > max_codes:
                unique_desc3 = unique_desc3.sample(n=max_codes, random_state=42)
                logger.warning(
                    f"ICD-10 context limited to {max_codes} random categories. "
                    f"Install sentence-transformers for better semantic matching."
                )
        except Exception as e:
            logger.warning(f"Semantic search failed: {e}, falling back to sampling")
            if len(unique_desc3) > max_codes:
                unique_desc3 = unique_desc3.sample(n=max_codes, random_state=42)

    elif len(unique_desc3) > max_codes:
        # Random sampling if semantic search not used
        unique_desc3 = unique_desc3.sample(n=max_codes, random_state=42)
        logger.warning(
            f"ICD-10 context limited to {max_codes} unique categories (description3). "
            f"Pass clinical_text parameter for semantic search across all categories."
        )

    # STAGE 2: Get all description2 values from the selected description3 categories
    selected_desc3_values = unique_desc3["description3"].unique()
    stage2_filtered_df = filtered_df[
        filtered_df["description3"].isin(selected_desc3_values)
    ]

    # Get unique description2 values for the context
    unique_desc2 = stage2_filtered_df.drop_duplicates(subset=["description2"])

    logger.info(
        f"Stage 2: Found {len(unique_desc2)} unique description2 values from {len(selected_desc3_values)} description3 categories"
    )

    # Build context string using description2 as the matching target
    context_lines = []
    for _, row in unique_desc2.iterrows():
        desc2 = row["description2"]
        desc3 = row["description3"]
        code = row["icd_fullcode"]
        # Format: description2 (Category: description3, Code: code)
        context_lines.append(f"{desc2} (Category: {desc3}, Code: {code})")

    return "\n".join(context_lines), stage2_filtered_df


def map_text_to_icd10(
    text: Union[str, List[str]],
    client: Optional[Any] = None,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    device: Optional[str] = None,
    category_filter: Optional[str] = None,
    max_context_codes: int = 500,
    include_reasoning: bool = True,
) -> pd.DataFrame:
    """
    Map clinical text to ICD-10 codes using instruction-following clinical LLM.

    This function uses OpenBioLLM (or other clinical LLMs) to analyze clinical
    text and map it to ICD-10 categories (description3), returning a DataFrame
    with all ICD-10 fields from the matched category.

    Args:
        text: Clinical text(s) to analyze - can be string or list of strings
        client: Pre-configured client (optional)
        model_name: Model identifier if client not provided (supports Ollama/HuggingFace/OpenAI)
        api_key: API key if using cloud provider
        device: Device for HuggingFace models ('cuda', 'cpu', or None for auto)
        category_filter: Optional ICD-10 category filter (e.g., 'J' for respiratory)
        max_context_codes: Maximum ICD-10 categories to include in prompt context
        include_reasoning: Whether to request reasoning for category selection

    Returns:
        DataFrame with columns:
        - original_text: The input clinical text
        - matched_description2: The matched ICD-10 disease/condition (from description2)
        - confidence: Confidence score (0-1) from LLM
        - reasoning: Explanation (if include_reasoning=True)
        - icd_cat, icd_code, icd_fullcode, description1, description2, description3, icd_code_who_eq:
          Fields from icd10_who.csv for the matched code

    Example:
        >>> # Default HuggingFace model (OpenBioLLM)
        >>> df = map_text_to_icd10(
        ...     "Patient presents with acute bronchitis",
        ...     device="cuda"  # or "cpu", will download model on first use
        ... )

        >>> # Explicit HuggingFace model
        >>> df = map_text_to_icd10(
        ...     "Type 2 diabetes mellitus without complications",
        ...     model_name="huggingface/aaditya/Llama3-OpenBioLLM-8B",
        ...     device="cuda"
        ... )

        >>> # Ollama model (if preferred)
        >>> df = map_text_to_icd10(
        ...     "Patient presents with acute bronchitis",
        ...     model_name="ollama/koesn/llama3-openbiollm-8b"
        ... )

        >>> print(df[['original_text', 'matched_description2', 'icd_fullcode']])

    Note:
        - Default uses HuggingFace model (downloads automatically on first use)
        - For Ollama: Ensure running with `ollama serve` and model pulled
        - For HuggingFace: Install transformers and torch
        - This is for research/educational purposes only - not for clinical use
    """
    # Convert single text to list for uniform processing
    if isinstance(text, str):
        texts = [text]
        single_input = True
    else:
        texts = text
        single_input = False

    # Get or create client
    if client is None:
        client = get_clinical_llm_client(
            model_name=model_name, api_key=api_key, device=device
        )

    # Adjust max_context_codes for HuggingFace models (smaller context window)
    adjusted_max_codes = max_context_codes
    if model_name and (
        model_name.startswith("huggingface/") or model_name.startswith("hf/")
    ):
        # HuggingFace models typically have 8K context, reduce to avoid truncation
        # 75 description3 categories → ~200-250 description2 codes (balance accuracy vs context)
        adjusted_max_codes = min(75, max_context_codes)
        logger.info(
            f"Reducing max_context_codes from {max_context_codes} to {adjusted_max_codes} "
            f"for HuggingFace model to avoid context length issues"
        )

    # Load ICD-10 reference data
    logger.info("Loading ICD-10 classification data...")
    icd10_df = load_icd10_data()

    # Process each text
    results = []

    for idx, clinical_text in enumerate(texts):
        logger.info(f"Mapping text {idx+1}/{len(texts)} to ICD-10...")

        # Build context for this specific clinical text (enables semantic search)
        icd10_context, filtered_icd10_df = build_icd10_context(
            icd10_df,
            category_filter=category_filter,
            max_codes=adjusted_max_codes,  # Use adjusted max codes
            clinical_text=clinical_text,  # Pass clinical text for semantic search
            use_semantic_search=True,
        )

        # Build prompt
        reasoning_instruction = (
            "\nProvide a brief reasoning explaining why this category was selected."
            if include_reasoning
            else ""
        )

        if include_reasoning:
            example_json = """
Example response format:
{
    "matched_description2": "Type 2 diabetes mellitus without complications",
    "confidence": 0.85,
    "reasoning": "Clinical text clearly indicates Type 2 diabetes without complications"
}"""
        else:
            example_json = """
Example response format:
{
    "matched_description2": "Type 2 diabetes mellitus without complications",
    "confidence": 0.85
}"""

        prompt = f"""You are an expert medical coder specializing in ICD-10 classification.

Your task is to analyze the following clinical text and match it to the most appropriate ICD-10 disease/condition.

Clinical Text:
{clinical_text}

Available ICD-10 Codes (description2 from WHO ICD-10):
{icd10_context}

Instructions:
1. Identify the primary diagnosis or condition in the clinical text
2. Select the MOST APPROPRIATE and SPECIFIC disease/condition from the reference list above
3. Return the EXACT description2 value as it appears in the reference list (matched_description2 field)
4. Provide a confidence score between 0.0 and 1.0 (confidence field){reasoning_instruction}

{example_json}

IMPORTANT:
- Use ONLY these field names: matched_description2, confidence{', reasoning' if include_reasoning else ''}
- matched_description2 MUST be an exact match from the reference list (the first part before " (Category:")
- Be conservative with confidence scores (0.7-0.9 for good matches, <0.7 for uncertain)
- If the text is ambiguous, reflect that in lower confidence scores
"""

        try:
            # Get structured output from LLM
            # Extract actual model name (remove provider prefix for API call)
            actual_model_name = model_name
            if model_name and model_name.startswith("ollama/"):
                actual_model_name = model_name.replace("ollama/", "")
            elif model_name and (
                model_name.startswith("huggingface/") or model_name.startswith("hf/")
            ):
                # For HF models, use the extracted name
                actual_model_name = model_name.replace("huggingface/", "").replace(
                    "hf/", ""
                )
            elif model_name and model_name.startswith("openai/"):
                actual_model_name = model_name.replace("openai/", "")

            response = client.chat.completions.create(
                model=actual_model_name or "aaditya/Llama3-OpenBioLLM-8B",
                response_model=ICD10Match,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert medical coder trained in ICD-10 classification. "
                        "You provide accurate disease category mappings.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )

            # Find matching ICD-10 records by description2
            normalized_description2 = response.matched_description2.strip()

            matched_rows = filtered_icd10_df[
                filtered_icd10_df["description2"].astype(str).str.strip()
                == normalized_description2
            ]

            if len(matched_rows) == 0:
                matched_rows = filtered_icd10_df[
                    filtered_icd10_df["description2"].astype(str).str.strip().str.casefold()
                    == normalized_description2.casefold()
                ]

            fallback_df = filtered_icd10_df
            if len(matched_rows) == 0:
                logger.warning(
                    f"No exact match for '{response.matched_description2}', attempting fuzzy match..."
                )

                candidate_descriptions = (
                    filtered_icd10_df["description2"].dropna().astype(str).str.strip().unique().tolist()
                )
                best_match = get_close_matches(
                    normalized_description2, candidate_descriptions, n=1, cutoff=0.75
                )

                if not best_match:
                    candidate_descriptions = (
                        icd10_df["description2"].dropna().astype(str).str.strip().unique().tolist()
                    )
                    best_match = get_close_matches(
                        normalized_description2, candidate_descriptions, n=1, cutoff=0.65
                    )
                    fallback_df = icd10_df

                if best_match:
                    logger.info(
                        "Fuzzy matched '%s' to '%s'",
                        normalized_description2,
                        best_match[0],
                    )
                    matched_rows = fallback_df[
                        fallback_df["description2"].astype(str).str.strip() == best_match[0]
                    ]

            semantic_candidates = _semantic_similarity_ranking(
                clinical_text, filtered_icd10_df
            )

            if len(matched_rows) > 0:
                match_row = matched_rows.iloc[0].to_dict()
                resolved_description2 = (
                    match_row.get("description2") or response.matched_description2
                )
                matched_similarity = None
                semantic_override = False

                if not semantic_candidates.empty:
                    semantic_top = semantic_candidates.iloc[0]
                    match_desc_casefold = resolved_description2.casefold()
                    candidate_match = semantic_candidates[
                        semantic_candidates["description2"].str.casefold()
                        == match_desc_casefold
                    ]
                    if not candidate_match.empty:
                        matched_similarity = float(candidate_match.iloc[0]["similarity"])
                    else:
                        matched_similarity = None

                    best_similarity = float(semantic_top["similarity"])
                    best_description = semantic_top["description2"]
                    similarity_gap = (
                        best_similarity
                        - (matched_similarity if matched_similarity is not None else -1.0)
                    )

                    if best_description.casefold() != match_desc_casefold and (
                        similarity_gap > 0.05 or float(response.confidence) < 0.5
                    ):
                        logger.info(
                            "Overriding LLM prediction '%s' with semantic match '%s' (similarity gap: %.3f)",
                            resolved_description2,
                            best_description,
                            similarity_gap,
                        )
                        match_row = semantic_top.to_dict()
                        resolved_description2 = match_row["description2"]
                        matched_similarity = best_similarity
                        semantic_override = True

                if matched_similarity is None and not semantic_candidates.empty:
                    matched_similarity = float(semantic_candidates.iloc[0]["similarity"])

                normalized_confidence = float(response.confidence)
                if matched_similarity is not None:
                    normalized_confidence = max(
                        normalized_confidence,
                        _normalize_cosine_similarity(matched_similarity),
                    )

                match_row["original_text"] = clinical_text
                match_row["matched_description2"] = resolved_description2
                match_row["confidence"] = normalized_confidence
                match_row["reasoning"] = (
                    response.reasoning if include_reasoning else None
                )
                results.append(match_row)

                logger.info(
                    "Matched to: %s (category: %s, code: %s, confidence: %.2f)%s",
                    resolved_description2,
                    match_row.get("description3", "N/A"),
                    match_row.get("icd_fullcode", "N/A"),
                    normalized_confidence,
                    " via semantic override" if semantic_override else "",
                )
            else:
                # No match found, create placeholder
                semantic_top_row = (
                    semantic_candidates.iloc[0] if not semantic_candidates.empty else None
                )

                if semantic_top_row is not None:
                    logger.info(
                        "Using semantic similarity fallback for '%s' -> '%s'",
                        clinical_text,
                        semantic_top_row["description2"],
                    )

                    match_row = semantic_top_row.to_dict()
                    similarity_score = float(semantic_top_row["similarity"])
                    match_row["original_text"] = clinical_text
                    match_row["matched_description2"] = match_row["description2"]
                    match_row["confidence"] = _normalize_cosine_similarity(
                        similarity_score
                    )
                    match_row["reasoning"] = (
                        response.reasoning if include_reasoning else None
                    )
                    results.append(match_row)

                    logger.info(
                        "Semantic fallback matched to: %s (code: %s, confidence: %.2f)",
                        match_row["description2"],
                        match_row.get("icd_fullcode", "N/A"),
                        match_row["confidence"],
                    )
                else:
                    logger.warning(f"Could not find ICD-10 match for: {clinical_text}")
                    results.append(
                        {
                            "original_text": clinical_text,
                            "matched_description2": response.matched_description2,
                            "confidence": response.confidence,
                            "reasoning": response.reasoning if include_reasoning else None,
                            "icd_cat": None,
                            "icd_code": None,
                            "icd_fullcode": None,
                            "description1": None,
                            "description2": None,
                            "description3": None,
                            "icd_code_who_eq": None,
                        }
                    )

        except Exception as e:
            logger.error(f"Failed to map text to ICD-10: {e}")
            # Create error placeholder
            results.append(
                {
                    "original_text": clinical_text,
                    "matched_description2": "ERROR",
                    "confidence": 0.0,
                    "reasoning": f"Error: {str(e)}",
                    "icd_cat": None,
                    "icd_code": None,
                    "icd_fullcode": None,
                    "description1": None,
                    "description2": None,
                    "description3": None,
                    "icd_code_who_eq": None,
                }
            )

    # Convert to DataFrame
    result_df = pd.DataFrame(results)

    # Reorder columns for better readability
    column_order = [
        "original_text",
        "matched_description2",
        "confidence",
        "icd_fullcode",
        "description1",
        "description2",
        "description3",
        "icd_cat",
        "icd_code",
        "icd_code_who_eq",
    ]
    if include_reasoning:
        column_order.insert(3, "reasoning")

    # Only include columns that exist
    column_order = [col for col in column_order if col in result_df.columns]
    result_df = result_df[column_order]

    return result_df


def batch_map_text_to_icd10(
    texts: List[str],
    client: Optional[Any] = None,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    device: Optional[str] = None,
    category_filter: Optional[str] = None,
    max_context_codes: int = 500,
    include_reasoning: bool = False,
) -> pd.DataFrame:
    """
    Map multiple clinical texts to ICD-10 codes in batch.

    This is a convenience wrapper around map_text_to_icd10() that processes
    a list of texts and returns a single DataFrame.

    Args:
        texts: List of clinical texts to analyze
        client: Pre-configured client (optional)
        model_name: Model identifier if client not provided (Ollama/HuggingFace/OpenAI)
        api_key: API key if using cloud provider
        device: Device for HuggingFace models ('cuda', 'cpu', or None for auto)
        category_filter: Optional ICD-10 category filter
        max_context_codes: Maximum ICD-10 categories to include in prompt context
        include_reasoning: Whether to request reasoning for category selection

    Returns:
        DataFrame with all mappings (same format as map_text_to_icd10)

    Example:
        >>> texts = [
        ...     "Type 2 diabetes mellitus",
        ...     "Essential hypertension",
        ... ]
        >>> df = batch_map_text_to_icd10(texts)
        >>> print(df[['original_text', 'matched_description3']])
    """
    # Simply call the main function with list of texts
    return map_text_to_icd10(
        text=texts,
        client=client,
        model_name=model_name,
        api_key=api_key,
        device=device,
        category_filter=category_filter,
        max_context_codes=max_context_codes,
        include_reasoning=include_reasoning,
    )
