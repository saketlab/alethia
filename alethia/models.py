import importlib.resources
import re
from typing import Any, Dict, List

import pandas as pd


def filter_huggingface_only(mteb_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter MTEB DataFrame to keep only HuggingFace models.

    Args:
        mteb_df: Original MTEB DataFrame

    Returns:
        pd.DataFrame: Filtered DataFrame containing only HuggingFace models
    """
    if mteb_df.empty:
        return mteb_df

    def is_huggingface_model(model_entry):
        if pd.isna(model_entry):
            return False

        model_str = str(model_entry).lower()

        if "huggingface.co" in model_str:
            return True

        match = re.search(r"\[(.*?)\]", model_entry)
        if match:
            clean_name = match.group(1)

            if (
                "/" in clean_name
                and not clean_name.startswith("models/")
                and not any(
                    api in clean_name.lower()
                    for api in ["gpt-", "text-embedding", "gemini-", "claude-"]
                )
            ):
                return True

        return False

    hf_mask = mteb_df["Model"].apply(is_huggingface_model)
    hf_df = mteb_df[hf_mask].copy()

    return hf_df


def load_mteb_dashboard_data() -> pd.DataFrame:
    """
    Load and process MTEB dashboard data from package resources.

    Returns:
        pd.DataFrame: Processed MTEB data with clean model names, memory in GB, and HuggingFace filtering applied
    """
    try:
        with importlib.resources.path(
            "alethia.data", "mteb_dashboard.csv"
        ) as data_path:
            if data_path is None:
                raise FileNotFoundError(
                    "MTEB dashboard data file path is None - package data not properly installed"
                )
            mteb_df = pd.read_csv(data_path)

        mteb_df = filter_huggingface_only(mteb_df)

        def extract_model_name(model_str):
            if pd.isna(model_str):
                return model_str

            # First try to extract from the markdown link format [text](url)
            match = re.search(r"\[(.*?)\]\((.*?)\)", str(model_str))
            if match:
                link_text = match.group(1)
                url = match.group(2)

                # Extract model name from HuggingFace URL if present
                # URLs like https://huggingface.co/Qwen/Qwen3-Embedding-8B
                hf_match = re.search(r"huggingface\.co/([^/?]+/[^/?]+)", url)
                if hf_match:
                    return hf_match.group(1)  # Returns "Qwen/Qwen3-Embedding-8B"

                # Fallback to link text if no HuggingFace URL pattern
                return link_text

            # Fallback to original behavior for [text] format
            match = re.search(r"\[(.*?)\]", str(model_str))
            return match.group(1) if match else str(model_str)

        mteb_df["clean_model_name"] = mteb_df["Model"].apply(extract_model_name)

        def convert_memory_to_gb(memory_str):
            if pd.isna(memory_str) or memory_str == "Unknown":
                return None
            try:
                return round(float(memory_str) / 1024, 2)
            except (ValueError, TypeError):
                return None

        mteb_df["memory_gb"] = mteb_df["Memory Usage (MB)"].apply(convert_memory_to_gb)

        mteb_df["clean_parameters"] = mteb_df["Number of Parameters"].fillna("Unknown")

        return mteb_df

    except Exception as e:
        print(f"Warning: Could not load MTEB data: {e}")
        return pd.DataFrame()


def classify_embedding_models() -> Dict[str, Dict[str, Any]]:
    """
    Classify embedding models into recommendation categories based on their characteristics.
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary mapping category names to model classifications with metadata
    """
    models_classification = {
        "lightweight": {
            "models": ["all-MiniLM-L6-v2", "nomic-embed-text", "phi3", "gemma"],
            "description": "Fast, lightweight models for speed-critical applications",
            "characteristics": {
                "memory_usage": "Low (< 200MB)",
                "inference_speed": "Very Fast",
                "quality": "Good",
                "dimensions": "384",
                "best_for": "Speed over quality",
            },
            "recommendations": {
                "when_to_use": "When you need fast inference and can accept slightly lower quality",
                "avoid_when": "Maximum quality is required",
            },
        },
        "fast_embedding": {
            "models": [
                "BAAI/bge-small-en-v1.5",
                "sentence-transformers/all-MiniLM-L6-v2",
                "snowflake/snowflake-arctic-embed-xs",
                "BAAI/bge-small-zh-v1.5",
                "jinaai/jina-embeddings-v2-small-en",
                "snowflake/snowflake-arctic-embed-s",
                "BAAI/bge-small-en",
                "nomic-ai/nomic-embed-text-v1.5-Q",
                "BAAI/bge-base-en-v1.5",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "Qdrant/clip-ViT-B-32-text",
            ],
            "description": "Ultra-fast, optimized embedding models prioritizing speed and efficiency",
            "characteristics": {
                "memory_usage": "Very Low (67MB - 250MB)",
                "inference_speed": "Extremely Fast",
                "quality": "Good to Very Good",
                "dimensions": "384-768",
                "best_for": "Real-time applications, high-throughput systems",
                "special_features": "ONNX optimized, quantized versions available, multilingual support",
            },
            "recommendations": {
                "when_to_use": "Real-time search, chatbots, mobile apps, high-volume processing, latency-critical applications",
                "avoid_when": "Research requiring highest possible accuracy, complex domain-specific tasks",
                "performance_notes": "Ideal for production systems where sub-second response times are critical",
            },
        },
        "high_performance": {
            "models": [
                "Salesforce/SFR-Embedding-Mistral",
                "sfr-embedding-mistral",
                "Salesforce/SFR-Embedding-2_R",
                "GritLM/GritLM-7B",
                "intfloat/e5-mistral-7b-instruct",
                "Alibaba-NLP/gte-Qwen2-7B-instruct",
                "qwen2",
                "deepseek-r1",
                "llama3.2",
                "mistral",
                "phi4",
            ],
            "description": "Large, high-quality models for maximum performance",
            "characteristics": {
                "memory_usage": "High (3-8GB)",
                "inference_speed": "Slower",
                "quality": "Excellent",
                "dimensions": "4096+",
                "best_for": "Maximum quality",
            },
            "recommendations": {
                "when_to_use": "When quality is paramount and you have sufficient compute resources",
                "avoid_when": "Speed is critical or limited resources",
            },
        },
        "balanced": {
            "models": [
                "mixedbread-ai/mxbai-embed-large-v1",
                "Linq-AI-Research/Linq-Embed-Mistral",
                "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
                "snowflake-arctic-embed",
                "gemma2",
            ],
            "description": "Balanced models offering good quality with reasonable resource usage",
            "characteristics": {
                "memory_usage": "Medium (1-3GB)",
                "inference_speed": "Moderate",
                "quality": "Very Good",
                "dimensions": "1024-2048",
                "best_for": "General purpose use",
            },
            "recommendations": {
                "when_to_use": "Default choice for most applications",
                "avoid_when": "Extreme speed or quality requirements",
            },
        },
        "specialized": {
            "models": ["nomic-ai/nomic-embed-text-v2-moe"],
            "description": "Models with specialized architectures (MoE, novel approaches)",
            "characteristics": {
                "memory_usage": "Variable",
                "inference_speed": "Variable",
                "quality": "High for specific tasks",
                "dimensions": "Variable",
                "best_for": "Specialized use cases",
            },
            "recommendations": {
                "when_to_use": "Experimental work or when architecture benefits are needed",
                "avoid_when": "Standard applications where proven models suffice",
            },
        },
    }
    return models_classification


def create_recommendation_matrix() -> Dict[str, List[str]]:
    """
    Create a recommendation matrix mapping use cases to appropriate model lists.
    Returns:
        Dict[str, List[str]]: Dictionary mapping use case names to lists of recommended models
    """
    recommendations = {
        "speed_critical": ["all-MiniLM-L6-v2", "phi3", "gemma", "nomic-embed-text"],
        "general_purpose": [
            "mixedbread-ai/mxbai-embed-large-v1",
            "Linq-AI-Research/Linq-Embed-Mistral",
            "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
            "snowflake-arctic-embed",
            "gemma2",
        ],
        "maximum_quality": [
            "Salesforce/SFR-Embedding-2_R",
            "Salesforce/SFR-Embedding-Mistral",
            "deepseek-r1",
            "Alibaba-NLP/gte-Qwen2-7B-instruct",
            "qwen2",
            "intfloat/e5-mistral-7b-instruct",
            "phi4",
            "mistral",
            "llama3.2",
        ],
        "research_experimental": [
            "nomic-ai/nomic-embed-text-v2-moe",
            "nomic-embed-text",
            "GritLM/GritLM-7B",
            "deepseek-r1",
        ],
        "production_balanced": [
            "mixedbread-ai/mxbai-embed-large-v1",
            "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
            "Linq-AI-Research/Linq-Embed-Mistral",
            "snowflake-arctic-embed",
            "gemma2",
        ],
        "memory_constrained": [
            "all-MiniLM-L6-v2",
            "phi3",
            "gemma",
            "nomic-ai/nomic-embed-text-v2-moe",
            "nomic-embed-text",
        ],
        "instruction_following": [
            "intfloat/multilingual-e5-large-instruct",
            "multilingual-e5-large-instruct",
            "Alibaba-NLP/gte-Qwen2-7B-instruct",
            "qwen2",
            "intfloat/e5-mistral-7b-instruct",
            "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
            "deepseek-r1",
        ],
        "efficient_architecture": [
            "nomic-ai/nomic-embed-text-v2-moe",
            "nomic-embed-text",
            "phi3",
            "phi4",
            "gemma2",
        ],
        "fast_embedding": [
            "BAAI/bge-small-en-v1.5",  # 0.067 GB, optimized
            "sentence-transformers/all-MiniLM-L6-v2",  # 0.09 GB, widely used
            "snowflake/snowflake-arctic-embed-xs",  # 0.09 GB, 2024
            "BAAI/bge-small-zh-v1.5",  # 0.09 GB, Chinese
            "jinaai/jina-embeddings-v2-small-en",  # 0.12 GB, 8192 tokens
            "snowflake/snowflake-arctic-embed-s",  # 0.13 GB, 2024
            "BAAI/bge-small-en",  # 0.13 GB, with prefixes
            "nomic-ai/nomic-embed-text-v1.5-Q",  # 0.13 GB, quantized
            "BAAI/bge-base-en-v1.5",  # 0.21 GB, good balance
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # 0.22 GB, multilingual
            "Qdrant/clip-ViT-B-32-text",  # 0.25 GB, multimodal
        ],
    }
    return recommendations


def get_detailed_model_info() -> Dict[str, Dict[str, Any]]:
    """
    Get detailed information for each embedding model including organization, size, parameters, and characteristics.

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary mapping model names to their detailed information
    """
    model_details = {
        "all-MiniLM-L6-v2": {
            "organization": "sentence-transformers",
            "size_category": "tiny",
            "estimated_params": "22M",
            "estimated_memory": "90MB",
            "dimensions": 384,
            "strengths": ["Very fast", "Lightweight", "Well-tested"],
            "weaknesses": ["Lower quality than larger models"],
            "best_use_case": "Speed-critical applications, prototyping",
        },
        "mixedbread-ai/mxbai-embed-large-v1": {
            "organization": "mixedbread-ai",
            "size_category": "large",
            "estimated_params": "335M",
            "estimated_memory": "1.3GB",
            "dimensions": 1024,
            "strengths": ["Strong performance", "Good balance", "Recent model"],
            "weaknesses": ["Larger than lightweight options"],
            "best_use_case": "General purpose, production applications",
        },
        "dragonkue/snowflake-arctic-embed-l-v2.0-ko": {
            "organization": "dragonkue",
            "size_category": "large",
            "estimated_params": "335M",
            "estimated_memory": "1.3GB",
            "dimensions": 1024,
            "strengths": ["Korean language optimization", "Multilingual"],
            "weaknesses": ["Specialized for Korean"],
            "best_use_case": "Korean language applications, Asian multilingual",
        },
        "snowflake-arctic-embed": {
            "organization": "Snowflake",
            "size_category": "large",
            "estimated_params": "335M",
            "estimated_memory": "1.3GB",
            "dimensions": 1024,
            "strengths": ["Good balance", "Production-ready", "Enterprise focus"],
            "weaknesses": ["Moderate resource usage"],
            "best_use_case": "Enterprise applications, balanced performance",
        },
        "nomic-ai/nomic-embed-text-v2-moe": {
            "organization": "nomic-ai",
            "size_category": "large",
            "estimated_params": "137M (MoE)",
            "estimated_memory": "600MB",
            "dimensions": 768,
            "strengths": [
                "Mixture of Experts",
                "Efficient architecture",
                "Good quality",
            ],
            "weaknesses": ["Complex architecture", "Less tested"],
            "best_use_case": "Research, efficient high-quality embeddings",
        },
        "nomic-embed-text": {
            "organization": "nomic-ai",
            "size_category": "medium",
            "estimated_params": "137M",
            "estimated_memory": "600MB",
            "dimensions": 768,
            "strengths": ["Efficient", "Good quality", "Open source"],
            "weaknesses": ["Newer model family"],
            "best_use_case": "Efficient embeddings, research applications",
        },
        "Linq-AI-Research/Linq-Embed-Mistral": {
            "organization": "Linq-AI-Research",
            "size_category": "large",
            "estimated_params": "400M",
            "estimated_memory": "1.6GB",
            "dimensions": 1024,
            "strengths": ["Mistral-based", "Strong performance", "Recent"],
            "weaknesses": ["Moderate resource usage"],
            "best_use_case": "High-quality general purpose embeddings",
        },
        "Alibaba-NLP/gte-Qwen2-7B-instruct": {
            "organization": "Alibaba-NLP",
            "size_category": "extra_large",
            "estimated_params": "7B",
            "estimated_memory": "7GB",
            "dimensions": 3584,
            "strengths": ["Excellent quality", "Instruction-tuned", "SOTA performance"],
            "weaknesses": ["Very large", "Slow inference", "High memory"],
            "best_use_case": "Maximum quality applications, research",
        },
        "qwen2": {
            "organization": "Alibaba-NLP",
            "size_category": "extra_large",
            "estimated_params": "1.5B-7B",
            "estimated_memory": "3-7GB",
            "dimensions": "1536-3584",
            "strengths": ["Strong performance", "Multiple sizes", "Instruction-tuned"],
            "weaknesses": ["Large variants require significant resources"],
            "best_use_case": "High-quality embeddings, various resource constraints",
        },
        "intfloat/multilingual-e5-large-instruct": {
            "organization": "intfloat",
            "size_category": "large",
            "estimated_params": "335M",
            "estimated_memory": "1.3GB",
            "dimensions": 1024,
            "strengths": ["Excellent multilingual", "Instruction-tuned", "Proven"],
            "weaknesses": ["Larger than monolingual alternatives"],
            "best_use_case": "Multilingual applications, cross-lingual tasks",
        },
        "multilingual-e5-large-instruct": {
            "organization": "intfloat",
            "size_category": "large",
            "estimated_params": "335M",
            "estimated_memory": "1.3GB",
            "dimensions": 1024,
            "strengths": ["Excellent multilingual", "Instruction-tuned", "Proven"],
            "weaknesses": ["Larger than monolingual alternatives"],
            "best_use_case": "Multilingual applications, cross-lingual tasks",
        },
        "Salesforce/SFR-Embedding-Mistral": {
            "organization": "Salesforce",
            "size_category": "extra_large",
            "estimated_params": "7B",
            "estimated_memory": "7GB",
            "dimensions": 4096,
            "strengths": [
                "Top MTEB performance",
                "Excellent quality",
                "Well-documented",
            ],
            "weaknesses": ["Very large", "Slow inference"],
            "best_use_case": "Maximum quality, research, high-stakes applications",
        },
        "sfr-embedding-mistral": {
            "organization": "Salesforce",
            "size_category": "extra_large",
            "estimated_params": "7B",
            "estimated_memory": "7GB",
            "dimensions": 4096,
            "strengths": [
                "Top MTEB performance",
                "Excellent quality",
                "Well-documented",
            ],
            "weaknesses": ["Very large", "Slow inference"],
            "best_use_case": "Maximum quality, research, high-stakes applications",
        },
        "GritLM/GritLM-7B": {
            "organization": "GritLM",
            "size_category": "extra_large",
            "estimated_params": "7B",
            "estimated_memory": "7GB",
            "dimensions": 4096,
            "strengths": ["Unified model", "Strong performance", "Novel approach"],
            "weaknesses": ["Very large", "Complex architecture"],
            "best_use_case": "Research, unified embedding and generation tasks",
        },
        "intfloat/e5-mistral-7b-instruct": {
            "organization": "intfloat",
            "size_category": "extra_large",
            "estimated_params": "7B",
            "estimated_memory": "7GB",
            "dimensions": 4096,
            "strengths": ["Excellent quality", "Instruction-tuned", "Proven family"],
            "weaknesses": ["Very large", "Slow inference"],
            "best_use_case": "High-quality embeddings, instruction-following tasks",
        },
        "Alibaba-NLP/gte-Qwen2-1.5B-instruct": {
            "organization": "Alibaba-NLP",
            "size_category": "medium",
            "estimated_params": "1.5B",
            "estimated_memory": "3GB",
            "dimensions": 1536,
            "strengths": ["Good balance", "Instruction-tuned", "Efficient"],
            "weaknesses": ["Moderate resource usage"],
            "best_use_case": "Balanced quality and efficiency",
        },
        "Lajavaness/bilingual-embedding-large": {
            "organization": "Lajavaness",
            "size_category": "large",
            "estimated_params": "335M",
            "estimated_memory": "1.3GB",
            "dimensions": 1024,
            "strengths": ["Bilingual optimization", "Specialized"],
            "weaknesses": ["Limited to specific language pairs"],
            "best_use_case": "Specific bilingual applications",
        },
        "Salesforce/SFR-Embedding-2_R": {
            "organization": "Salesforce",
            "size_category": "extra_large",
            "estimated_params": "7B",
            "estimated_memory": "7GB",
            "dimensions": 4096,
            "strengths": [
                "Latest Salesforce model",
                "Improved performance",
                "High quality",
            ],
            "weaknesses": ["Very large", "Slow inference"],
            "best_use_case": "Latest high-quality embeddings, research",
        },
        "deepseek-r1": {
            "organization": "DeepSeek",
            "size_category": "extra_large",
            "estimated_params": "7B-67B",
            "estimated_memory": "7-67GB",
            "dimensions": 4096,
            "strengths": ["Advanced reasoning", "Latest model", "Strong performance"],
            "weaknesses": ["Very large", "High resource requirements"],
            "best_use_case": "Advanced reasoning tasks, research applications",
        },
        "llama3.2": {
            "organization": "Meta",
            "size_category": "extra_large",
            "estimated_params": "1B-90B",
            "estimated_memory": "1-90GB",
            "dimensions": "2048-8192",
            "strengths": ["Strong base model", "Multiple sizes", "Well-supported"],
            "weaknesses": ["Large variants require significant resources"],
            "best_use_case": "General purpose, various scales available",
        },
        "gemma": {
            "organization": "Google",
            "size_category": "small",
            "estimated_params": "2B-7B",
            "estimated_memory": "2-7GB",
            "dimensions": "2048-3072",
            "strengths": ["Lightweight", "Google quality", "Efficient"],
            "weaknesses": ["Smaller than top performers"],
            "best_use_case": "Lightweight applications, edge deployment",
        },
        "gemma2": {
            "organization": "Google",
            "size_category": "medium",
            "estimated_params": "2B-27B",
            "estimated_memory": "2-27GB",
            "dimensions": "2304-4608",
            "strengths": ["Improved over Gemma", "Good balance", "Latest Google"],
            "weaknesses": ["Moderate resource usage"],
            "best_use_case": "Balanced performance, production applications",
        },
        "phi3": {
            "organization": "Microsoft",
            "size_category": "small",
            "estimated_params": "3.8B",
            "estimated_memory": "4GB",
            "dimensions": 3072,
            "strengths": ["Efficient", "Small but capable", "Microsoft quality"],
            "weaknesses": ["Limited compared to larger models"],
            "best_use_case": "Efficient deployments, edge computing",
        },
        "phi4": {
            "organization": "Microsoft",
            "size_category": "extra_large",
            "estimated_params": "14B",
            "estimated_memory": "14GB",
            "dimensions": 5120,
            "strengths": [
                "Latest Phi model",
                "Strong performance",
                "Efficient architecture",
            ],
            "weaknesses": ["Large size"],
            "best_use_case": "High-quality applications, latest Microsoft tech",
        },
        "mistral": {
            "organization": "Mistral AI",
            "size_category": "extra_large",
            "estimated_params": "7B-22B",
            "estimated_memory": "7-22GB",
            "dimensions": "4096-8192",
            "strengths": ["Strong performance", "European model", "Good quality"],
            "weaknesses": ["Large resource requirements"],
            "best_use_case": "High-quality embeddings, European preference",
        },
    }

    return model_details


def print_model_classification_guide():
    """
    Print a comprehensive model classification and recommendation guide to console.
    """
    print("🎯 Embedding Model Classification & Recommendation Guide")
    print("=" * 70)

    classifications = classify_embedding_models()
    model_details = get_detailed_model_info()
    recommendations = create_recommendation_matrix()

    for category, info in classifications.items():
        print(f"\n📂 {category.upper().replace('_', ' ')} CATEGORY")
        print(f"Description: {info['description']}")
        print(
            f"Characteristics: {info['characteristics']['memory_usage']}, {info['characteristics']['quality']} quality"
        )
        print(f"Best for: {info['characteristics']['best_for']}")

        print("\nModels in this category:")
        for model in info["models"]:
            details = model_details.get(model, {})
            memory = details.get("estimated_memory", "Unknown")
            dims = details.get("dimensions", "Unknown")
            print(f"  • {model}")
            print(
                f"    Memory: {memory} | Dimensions: {dims} | Use: {details.get('best_use_case', 'General')}"
            )

        print(f"When to use: {info['recommendations']['when_to_use']}")
        print(f"Avoid when: {info['recommendations']['avoid_when']}")
        print("-" * 50)

    print("\n🎯 RECOMMENDATIONS BY USE CASE")
    print("=" * 40)

    use_case_descriptions = {
        "speed_critical": "When inference speed is the top priority",
        "general_purpose": "Default choice for most applications",
        "maximum_quality": "When you need the highest possible quality",
        "production_balanced": "Proven models for production deployment",
        "memory_constrained": "When memory/compute resources are limited",
        "instruction_following": "Models trained to follow instructions better",
        "efficient_architecture": "Models with optimized architectures",
    }

    for use_case, models in recommendations.items():
        description = use_case_descriptions.get(use_case, "")
        print(f"\n{use_case.replace('_', ' ').title()}: {description}")
        for i, model in enumerate(models, 1):
            details = model_details.get(model, {})
            memory = details.get("estimated_memory", "Unknown")
            print(f"  {i}. {model} ({memory})")


def get_model_recommendation(use_case: str, constraint: str = None) -> List[str]:
    """
    Get model recommendations for a specific use case with optional constraints.

    Args:
        use_case: The use case for which to get recommendations (e.g., 'speed_critical', 'maximum_quality')
        constraint: Optional constraint to apply ('low_memory', 'fast_inference', or None)

    Returns:
        List[str]: List of recommended model names for the specified use case and constraints
    """
    recommendations = create_recommendation_matrix()

    if use_case not in recommendations:
        return []

    models = recommendations[use_case].copy()

    if constraint == "low_memory":
        model_details = get_detailed_model_info()
        models = [
            m
            for m in models
            if model_details.get(m, {}).get("size_category")
            in ["tiny", "small", "medium"]
        ]

    elif constraint == "fast_inference":
        model_details = get_detailed_model_info()
        models.sort(
            key=lambda m: {
                "tiny": 0,
                "small": 1,
                "medium": 2,
                "large": 3,
                "extra_large": 4,
            }.get(model_details.get(m, {}).get("size_category", "large"), 3)
        )

    return models


def get_medical_models() -> Dict[str, Dict[str, Any]]:
    """
    Get curated list of biomedical/clinical embedding models suitable for medical NLP tasks.
    All models have ≤8B parameters and are optimized for medical/clinical text processing.

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary mapping model names to their detailed information
    """
    medical_models = {
        # Small Language Models (SLMs) - Prioritized for efficiency
        "NeuML/pubmedbert-base-embeddings": {
            "organization": "NeuML",
            "base_model": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
            "size_category": "small",
            "parameters": "109.5M",
            "memory_gb": 0.4,
            "embedding_dimension": 768,
            "model_type": "sentence-transformer",
            "training_data": "PubMed abstracts and full-text articles",
            "strengths": [
                "Purpose-built sentence-transformer architecture for biomedical retrieval",
                "Optimized for medical terminology, disease names, and pathogens",
                "Widely adopted across clinical chatbots and medical search systems",
                "High quality embeddings for medical text",
            ],
            "use_cases": [
                "Medical literature search",
                "Clinical document similarity",
                "Biomedical entity matching",
                "SNOMED-ICD mapping",
            ],
            "benchmark_performance": {
                "pubmed_qa": "95.62%",
                "medical_text_similarity": "excellent",
            },
            "loading_method": "sentence-transformers",
            "recommended": True,
            "priority": 1,  # Highest priority - smallest and highly effective
        },
        "emilyalsentzer/Bio_ClinicalBERT": {
            "organization": "emilyalsentzer",
            "base_model": "BioBERT",
            "size_category": "small",
            "parameters": "110M",
            "memory_gb": 0.4,
            "embedding_dimension": 768,
            "model_type": "transformer",
            "training_data": "MIMIC-III clinical notes (880M words)",
            "strengths": [
                "Clinical notes specialization",
                "EHR optimization",
                "Well-established model",
            ],
            "use_cases": [
                "Clinical notes processing",
                "EHR text analysis",
                "Clinical terminology matching",
            ],
            "loading_method": "transformers",
            "recommended": True,
            "priority": 2,
        },
        "medicalai/ClinicalBERT": {
            "organization": "medicalai",
            "base_model": "BERT",
            "size_category": "small",
            "parameters": "110M",
            "memory_gb": 0.4,
            "embedding_dimension": 768,
            "model_type": "transformer",
            "training_data": "1.2B words from diverse disease corpora + 3M patient EHR records",
            "strengths": [
                "Trained on large medical dataset (1.2B words)",
                "Fine-tuned on 3M+ patient EHRs",
                "Diverse disease corpora coverage",
                "Masked language modeling pre-training",
            ],
            "use_cases": [
                "Medical NLP tasks",
                "Disease diagnosis assistance",
                "Medical text analysis",
                "Fill-mask operations in medical context",
            ],
            "loading_method": "transformers",
            "recommended": True,
            "priority": 2,
        },
        "pritamdeka/S-PubMedBert-MS-MARCO": {
            "organization": "pritamdeka",
            "base_model": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
            "size_category": "small",
            "parameters": "110M",
            "memory_gb": 0.4,
            "embedding_dimension": 768,
            "model_type": "sentence-transformer",
            "training_data": "PubMed + MS-MARCO",
            "strengths": [
                "Information retrieval optimized",
                "Medical domain + IR training",
                "Sentence-transformer ready",
            ],
            "use_cases": [
                "Medical information retrieval",
                "Clinical document search",
                "Biomedical Q&A",
            ],
            "loading_method": "sentence-transformers",
            "recommended": True,
            "priority": 2,
        },
        "FremyCompany/BioLORD-2023": {
            "organization": "FremyCompany",
            "base_model": "MPNet",
            "size_category": "small",
            "parameters": "109M",
            "memory_gb": 0.4,
            "embedding_dimension": 768,
            "model_type": "sentence-transformer",
            "training_data": "Biomedical datasets, EHR records",
            "strengths": [
                "Semantic search optimized",
                "Clinical notes specialization",
                "MPNet architecture efficiency",
            ],
            "use_cases": [
                "EHR document processing",
                "Clinical semantic search",
                "Medical document clustering",
            ],
            "loading_method": "sentence-transformers",
            "recommended": True,
            "priority": 2,
        },
        "lokeshch19/ModernPubMedBERT": {
            "organization": "lokeshch19",
            "base_model": "PubMedBERT",
            "size_category": "small",
            "parameters": "149M",
            "memory_gb": 0.6,
            "embedding_dimension": 768,
            "model_type": "sentence-transformer",
            "training_data": "PubMed title-abstract pairs with InfoNCE",
            "strengths": [
                "Modern contrastive learning",
                "Long context (2048 tokens)",
                "Enhanced medical concept understanding",
                "Superior medical vs non-medical discrimination",
            ],
            "use_cases": [
                "Medical document similarity",
                "Long clinical documents",
                "Medical concept clustering",
            ],
            "loading_method": "sentence-transformers",
            "recommended": True,
            "priority": 2,
        },
        "UFNLP/gatortron-base": {
            "organization": "UFNLP",
            "base_model": "BERT-Megatron",
            "size_category": "medium",
            "parameters": "345M",
            "memory_gb": 1.4,
            "embedding_dimension": 768,
            "model_type": "transformer",
            "training_data": "82B words clinical notes + PubMed + WikiText",
            "strengths": [
                "Massive clinical training data",
                "MIMIC-III + PubMed",
                "Strong clinical NLP performance",
            ],
            "use_cases": [
                "Clinical NER",
                "Relation extraction",
                "Clinical text understanding",
            ],
            "loading_method": "transformers",
            "recommended": True,
            "priority": 3,
        },
        "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext": {
            "organization": "microsoft",
            "base_model": "BERT",
            "size_category": "small",
            "parameters": "110M",
            "memory_gb": 0.4,
            "embedding_dimension": 768,
            "model_type": "transformer",
            "training_data": "PubMed abstracts and full-text",
            "strengths": [
                "Domain-specific vocabulary",
                "Pre-trained from scratch on biomedical text",
                "Top BLURB benchmark performance",
            ],
            "use_cases": [
                "Biomedical NLP tasks",
                "Medical text classification",
                "Base model for fine-tuning",
            ],
            "loading_method": "transformers",
            "recommended": True,
            "priority": 2,
        },
        "gsarti/biobert-nli": {
            "organization": "gsarti",
            "base_model": "BioBERT",
            "size_category": "small",
            "parameters": "110M",
            "memory_gb": 0.4,
            "embedding_dimension": 768,
            "model_type": "sentence-transformer",
            "training_data": "BioBERT + SNLI + MultiNLI",
            "strengths": [
                "NLI fine-tuning for better sentence embeddings",
                "Universal sentence embeddings",
                "Biomedical domain knowledge",
            ],
            "use_cases": [
                "Sentence similarity",
                "Semantic textual similarity",
                "Medical paraphrase detection",
            ],
            "loading_method": "sentence-transformers",
            "recommended": True,
            "priority": 2,
        },
        "cambridgeltl/SapBERT-from-PubMedBERT-fulltext": {
            "organization": "cambridgeltl",
            "base_model": "PubMedBERT",
            "size_category": "small",
            "parameters": "110M",
            "memory_gb": 0.4,
            "embedding_dimension": 768,
            "model_type": "sentence-transformer",
            "training_data": "PubMedBERT + UMLS (self-alignment pretraining)",
            "strengths": [
                "Self-alignment pretraining on UMLS",
                "Optimized for medical entity linking",
                "Strong biomedical entity representation",
                "Specialized for SNOMED/ICD mapping tasks",
            ],
            "use_cases": [
                "Medical entity linking",
                "SNOMED-ICD mapping",
                "Biomedical entity normalization",
                "Clinical terminology alignment",
            ],
            "loading_method": "sentence-transformers",
            "recommended": True,
            "priority": 1,  # High priority for entity mapping tasks
        },
        "sentence-transformers/embeddinggemma-300m-medical": {
            "organization": "sentence-transformers",
            "base_model": "Google EmbeddingGemma",
            "size_category": "medium",
            "parameters": "302.9M",
            "memory_gb": 1.2,
            "embedding_dimension": 1024,
            "model_type": "sentence-transformer",
            "training_data": "MIRIAD medical dataset (100K samples) with CachedMultipleNegativesRankingLoss",
            "strengths": [
                "Medical-specific fine-tune of EmbeddingGemma",
                "Modern architecture with active maintenance",
                "Optimized for clinical terminology and retrieval",
            ],
            "use_cases": [
                "Clinical semantic search",
                "Medical terminology retrieval",
                "Biomedical question answering",
                "Modern clinical AI applications",
            ],
            "loading_method": "sentence-transformers",
            "recommended": True,
            "priority": 2,
        },
        "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb": {
            "organization": "pritamdeka",
            "base_model": "BioBERT",
            "size_category": "small",
            "parameters": "110M",
            "memory_gb": 0.4,
            "embedding_dimension": 768,
            "model_type": "sentence-transformer",
            "training_data": "BioBERT fine-tuned on MNLI, SNLI, SciNLI, SciTail, MedNLI, and STS-B",
            "strengths": [
                "Biomedical sentence-transformer optimized for semantic similarity",
                "Trained on medical NLI datasets including MedNLI",
                "Strong cross-domain generalization within biomedical text",
            ],
            "use_cases": [
                "Disease and symptom similarity",
                "Clinical paraphrase detection",
                "Medical terminology alignment",
                "Biomedical sentence inference",
            ],
            "loading_method": "sentence-transformers",
            "recommended": True,
            "priority": 2,
        },
        "menadsa/S-Bio_ClinicalBERT": {
            "organization": "menadsa",
            "base_model": "Bio_ClinicalBERT",
            "size_category": "small",
            "parameters": "110M",
            "memory_gb": 0.4,
            "embedding_dimension": 768,
            "model_type": "sentence-transformer",
            "training_data": "Clinical notes from electronic health records",
            "strengths": [
                "Optimized for semantic similarity on clinical documentation",
                "Captures clinical abbreviations and shorthand",
                "Retains Bio_ClinicalBERT knowledge with sentence-transformer pooling",
            ],
            "use_cases": [
                "Clinical note similarity",
                "Treatment and procedure comparison",
                "EHR semantic search",
                "Care pathway clustering",
            ],
            "loading_method": "sentence-transformers",
            "recommended": True,
            "priority": 2,
        },
        "pritamdeka/SapBERT-mnli-snli-scinli-scitail-mednli-stsb": {
            "organization": "pritamdeka",
            "base_model": "SapBERT",
            "size_category": "small",
            "parameters": "110M",
            "memory_gb": 0.4,
            "embedding_dimension": 768,
            "model_type": "sentence-transformer",
            "training_data": "SapBERT fine-tuned on MNLI, SNLI, SciNLI, SciTail, MedNLI, and STS-B",
            "strengths": [
                "Combines entity linking strength of SapBERT with sentence-level supervision",
                "Excels at medical synonym matching and semantic similarity",
                "Supports both entity-level and sentence-level embeddings",
            ],
            "use_cases": [
                "Medical entity linking",
                "Clinical terminology normalization",
                "Biomedical semantic search",
                "Ontology-aware similarity",
            ],
            "loading_method": "sentence-transformers",
            "recommended": True,
            "priority": 1,
        },
        "dmis-lab/biosyn-sapbert-bc5cdr-disease": {
            "organization": "dmis-lab",
            "base_model": "SapBERT",
            "size_category": "small",
            "parameters": "109M",
            "memory_gb": 0.4,
            "embedding_dimension": 768,
            "model_type": "sentence-transformer",
            "training_data": "BioSyn SapBERT fine-tuned on BC5CDR disease corpus",
            "strengths": [
                "Specialized for disease entity embedding and normalization",
                "High accuracy on disease mention detection tasks",
                "Maintains SapBERT alignment across medical vocabularies",
            ],
            "use_cases": [
                "Disease mention normalization",
                "Medical ontology mapping",
                "Clinical diagnosis clustering",
                "Disease terminology search",
            ],
            "loading_method": "sentence-transformers",
            "recommended": True,
            "priority": 1,
        },
        "microsoft/BiomedNLP-KRISSBERT-PubMed-UMLS-EL": {
            "organization": "microsoft",
            "base_model": "KRISSBERT",
            "size_category": "small",
            "parameters": "110M",
            "memory_gb": 0.4,
            "embedding_dimension": 768,
            "model_type": "transformer",
            "training_data": "PubMed literature combined with UMLS knowledge-rich synonym sets",
            "strengths": [
                "Designed for biomedical entity linking with UMLS integration",
                "Captures synonym sets across standardized medical vocabularies",
                "Strong performance on concept normalization tasks",
            ],
            "use_cases": [
                "UMLS and SNOMED concept linking",
                "Medical terminology normalization",
                "Clinical knowledge graph integration",
            ],
            "loading_method": "transformers",
            "recommended": True,
            "priority": 1,
        },
        "dmis-lab/biobert-v1.1": {
            "organization": "dmis-lab",
            "base_model": "BERT",
            "size_category": "small",
            "parameters": "110M",
            "memory_gb": 0.4,
            "embedding_dimension": 768,
            "model_type": "transformer",
            "training_data": "PubMed abstracts and PMC full-text articles",
            "strengths": [
                "Original BioBERT pretrained on large biomedical corpora",
                "Widely validated baseline for biomedical NLP",
                "Strong foundation for task-specific fine-tuning",
            ],
            "use_cases": [
                "Feature extraction for biomedical tasks",
                "Baseline biomedical embeddings",
                "Domain-specific fine-tuning",
            ],
            "loading_method": "transformers",
            "recommended": True,
            "priority": 2,
        },
        "dmis-lab/biobert-base-cased-v1.1": {
            "organization": "dmis-lab",
            "base_model": "BERT",
            "size_category": "small",
            "parameters": "110M",
            "memory_gb": 0.4,
            "embedding_dimension": 768,
            "model_type": "transformer",
            "training_data": "PubMed abstracts and PMC full-text articles (cased)",
            "strengths": [
                "Preserves capitalization for medical acronyms and names",
                "Reliable biomedical baseline",
                "Supports case-sensitive terminology matching",
            ],
            "use_cases": [
                "Medical acronym resolution",
                "Case-sensitive terminology extraction",
                "Biomedical classification tasks",
            ],
            "loading_method": "transformers",
            "recommended": True,
            "priority": 2,
        },
        "dmis-lab/biobert-base-cased-v1.2": {
            "organization": "dmis-lab",
            "base_model": "BERT",
            "size_category": "small",
            "parameters": "110M",
            "memory_gb": 0.4,
            "embedding_dimension": 768,
            "model_type": "transformer",
            "training_data": "Updated BioBERT cased pre-training on PubMed and PMC",
            "strengths": [
                "Improved training procedure over v1.1",
                "Maintains case sensitivity for medical abbreviations",
                "Strong baseline for biomedical downstream tasks",
            ],
            "use_cases": [
                "Biomedical text classification",
                "Terminology preservation",
                "Fine-tuning for clinical NLP",
            ],
            "loading_method": "transformers",
            "recommended": True,
            "priority": 2,
        },
        "dmis-lab/biobert-large-cased-v1.1": {
            "organization": "dmis-lab",
            "base_model": "BERT",
            "size_category": "medium",
            "parameters": "340M",
            "memory_gb": 1.3,
            "embedding_dimension": 1024,
            "model_type": "transformer",
            "training_data": "PubMed abstracts and PMC full-text articles",
            "strengths": [
                "Higher capacity BioBERT for complex medical semantics",
                "Improved performance on nuanced biomedical relationships",
                "Handles multi-concept medical contexts",
            ],
            "use_cases": [
                "Advanced biomedical reasoning",
                "Complex entity relationship modeling",
                "High-accuracy medical NLP",
            ],
            "loading_method": "transformers",
            "recommended": True,
            "priority": 3,
        },
        "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext": {
            "organization": "microsoft",
            "base_model": "BiomedBERT",
            "size_category": "small",
            "parameters": "110M",
            "memory_gb": 0.4,
            "embedding_dimension": 768,
            "model_type": "transformer",
            "training_data": "PubMed abstracts and PMC full-text articles",
            "strengths": [
                "Trained from scratch on biomedical literature",
                "Comprehensive coverage of medical terminology",
                "Strong performance across biomedical benchmarks",
            ],
            "use_cases": [
                "Biomedical NLP tasks",
                "Medical text classification",
                "Base model for fine-tuning",
            ],
            "loading_method": "transformers",
            "recommended": True,
            "priority": 2,
        },
        "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract": {
            "organization": "microsoft",
            "base_model": "BiomedBERT",
            "size_category": "small",
            "parameters": "110M",
            "memory_gb": 0.4,
            "embedding_dimension": 768,
            "model_type": "transformer",
            "training_data": "PubMed abstracts",
            "strengths": [
                "Faster training focus on abstract-level terminology",
                "Maintains strong biomedical vocabulary coverage",
                "Efficient base for biomedical tasks",
            ],
            "use_cases": [
                "Abstract-focused biomedical analysis",
                "Medical concept extraction",
                "Fine-tuning for lightweight biomedical models",
            ],
            "loading_method": "transformers",
            "recommended": True,
            "priority": 2,
        },
        "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract": {
            "organization": "microsoft",
            "base_model": "BiomedBERT",
            "size_category": "medium",
            "parameters": "340M",
            "memory_gb": 1.3,
            "embedding_dimension": 1024,
            "model_type": "transformer",
            "training_data": "PubMed abstracts",
            "strengths": [
                "Higher capacity embeddings for complex biomedical relationships",
                "Improved performance on rare terminology",
                "Supports nuanced biomedical inference",
            ],
            "use_cases": [
                "Advanced biomedical understanding",
                "Rare disease terminology modeling",
                "High-accuracy biomedical search",
            ],
            "loading_method": "transformers",
            "recommended": True,
            "priority": 3,
        },
        "nlpie/distil-clinicalbert": {
            "organization": "nlpie",
            "base_model": "Bio_ClinicalBERT",
            "size_category": "tiny",
            "parameters": "66M",
            "memory_gb": 0.25,
            "embedding_dimension": 768,
            "model_type": "transformer",
            "training_data": "Distilled ClinicalBERT trained on clinical EHR corpora",
            "strengths": [
                "Efficient clinical embeddings with reduced footprint",
                "Retains key clinical terminology understanding",
                "Suitable for latency-sensitive deployments",
            ],
            "use_cases": [
                "Real-time clinical text processing",
                "Resource-constrained medical applications",
                "Clinical similarity search at scale",
            ],
            "loading_method": "transformers",
            "recommended": True,
            "priority": 2,
        },
        "nlpie/tiny-clinicalbert": {
            "organization": "nlpie",
            "base_model": "Bio_ClinicalBERT",
            "size_category": "tiny",
            "parameters": "30M",
            "memory_gb": 0.1,
            "embedding_dimension": 312,
            "model_type": "transformer",
            "training_data": "Tiny ClinicalBERT distilled from Bio_ClinicalBERT for EHR data",
            "strengths": [
                "Ultra-compact clinical model for edge and mobile scenarios",
                "Maintains core clinical vocabulary understanding",
                "Very fast inference for streaming workloads",
            ],
            "use_cases": [
                "Edge deployment of clinical NLP",
                "High-throughput clinical document embedding",
                "Latency-critical healthcare applications",
            ],
            "loading_method": "transformers",
            "recommended": True,
            "priority": 1,
        },
        "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12": {
            "organization": "bionlp",
            "base_model": "BlueBERT",
            "size_category": "small",
            "parameters": "110M",
            "memory_gb": 0.4,
            "embedding_dimension": 768,
            "model_type": "transformer",
            "training_data": "PubMed abstracts combined with MIMIC-III clinical notes",
            "strengths": [
                "Bridges biomedical literature and clinical documentation",
                "Captures terminology across research and practice",
                "Reliable dual-domain embeddings",
            ],
            "use_cases": [
                "Cross-domain medical text analysis",
                "Clinical and literature terminology alignment",
                "Biomedical search spanning multiple sources",
            ],
            "loading_method": "transformers",
            "recommended": True,
            "priority": 2,
        },
        "bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12": {
            "organization": "bionlp",
            "base_model": "BlueBERT",
            "size_category": "small",
            "parameters": "110M",
            "memory_gb": 0.4,
            "embedding_dimension": 768,
            "model_type": "transformer",
            "training_data": "PubMed abstracts",
            "strengths": [
                "Focused on biomedical literature terminology",
                "Good baseline for research text embeddings",
                "Maintains BlueBERT efficiency",
            ],
            "use_cases": [
                "Biomedical literature search",
                "Scientific terminology embedding",
                "Research paper similarity",
            ],
            "loading_method": "transformers",
            "recommended": True,
            "priority": 2,
        },
        "bionlp/bluebert_pubmed_mimic_uncased_L-24_H-1024_A-16": {
            "organization": "bionlp",
            "base_model": "BlueBERT",
            "size_category": "medium",
            "parameters": "340M",
            "memory_gb": 1.3,
            "embedding_dimension": 1024,
            "model_type": "transformer",
            "training_data": "PubMed abstracts and MIMIC-III clinical notes",
            "strengths": [
                "Large dual-domain model with 1024-d embeddings",
                "Superior capacity for complex medical relationships",
                "Handles detailed clinical and research terminology",
            ],
            "use_cases": [
                "High-accuracy cross-domain retrieval",
                "Comprehensive medical concept representation",
                "Advanced clinical decision support",
            ],
            "loading_method": "transformers",
            "recommended": True,
            "priority": 3,
        },
        "bionlp/bluebert_pubmed_uncased_L-24_H-1024_A-16": {
            "organization": "bionlp",
            "base_model": "BlueBERT",
            "size_category": "medium",
            "parameters": "340M",
            "memory_gb": 1.3,
            "embedding_dimension": 1024,
            "model_type": "transformer",
            "training_data": "PubMed abstracts",
            "strengths": [
                "Large-capacity model for biomedical literature",
                "Improved performance on nuanced scientific terminology",
                "Provides 1024-d dimension embeddings",
            ],
            "use_cases": [
                "Detailed biomedical literature analysis",
                "Complex terminology retrieval",
                "Research-focused semantic search",
            ],
            "loading_method": "transformers",
            "recommended": True,
            "priority": 3,
        },
        "allenai/scibert_scivocab_uncased": {
            "organization": "allenai",
            "base_model": "SciBERT",
            "size_category": "small",
            "parameters": "110M",
            "memory_gb": 0.4,
            "embedding_dimension": 768,
            "model_type": "transformer",
            "training_data": "1.14M Semantic Scholar papers (82% biomedical, 18% CS)",
            "strengths": [
                "Scientific vocabulary tailored for technical terminology",
                "Strong performance on biomedical literature",
                "Widely adopted scientific baseline",
            ],
            "use_cases": [
                "Biomedical literature understanding",
                "Scientific document embedding",
                "Medical concept extraction in research papers",
            ],
            "loading_method": "transformers",
            "recommended": True,
            "priority": 3,
        },
        "allenai/scibert_scivocab_cased": {
            "organization": "allenai",
            "base_model": "SciBERT",
            "size_category": "small",
            "parameters": "110M",
            "memory_gb": 0.4,
            "embedding_dimension": 768,
            "model_type": "transformer",
            "training_data": "1.14M Semantic Scholar papers with cased vocabulary",
            "strengths": [
                "Preserves capitalization for scientific acronyms and names",
                "Handles biomedical terminology with case sensitivity",
                "Strong baseline for scientific NLP",
            ],
            "use_cases": [
                "Case-sensitive biomedical text analysis",
                "Scientific acronym handling",
                "Research literature semantic search",
            ],
            "loading_method": "transformers",
            "recommended": True,
            "priority": 3,
        },
        "heycao/scibert_scivocab_cased_sentence": {
            "organization": "heycao",
            "base_model": "SciBERT",
            "size_category": "small",
            "parameters": "109.9M",
            "memory_gb": 0.4,
            "embedding_dimension": 768,
            "model_type": "sentence-transformer",
            "training_data": "SciBERT fine-tuned for sentence embeddings (2025)",
            "strengths": [
                "Sentence-transformer optimized SciBERT",
                "Maintains scientific vocabulary while improving sentence pooling",
                "Recent model tuned for scientific sentence similarity",
            ],
            "use_cases": [
                "Biomedical sentence similarity",
                "Scientific abstract retrieval",
                "Medical fact alignment",
            ],
            "loading_method": "sentence-transformers",
            "recommended": True,
            "priority": 2,
        },
        "bvanaken/CORe-clinical-outcome-biobert-v1": {
            "organization": "bvanaken",
            "base_model": "BioBERT",
            "size_category": "small",
            "parameters": "108.3M",
            "memory_gb": 0.4,
            "embedding_dimension": 768,
            "model_type": "sentence-transformer",
            "training_data": "Clinical outcome representations with patient trajectories",
            "strengths": [
                "Captures clinical outcomes and temporal signals",
                "Designed for patient trajectory understanding",
                "Retains biomedical vocabulary from BioBERT",
            ],
            "use_cases": [
                "Outcome prediction feature extraction",
                "Patient similarity analysis",
                "Clinical cohort discovery",
            ],
            "loading_method": "sentence-transformers",
            "recommended": True,
            "priority": 2,
        },
        "pritamdeka/S-Scibert-snli-multinli-stsb": {
            "organization": "pritamdeka",
            "base_model": "SciBERT",
            "size_category": "small",
            "parameters": "110M",
            "memory_gb": 0.4,
            "embedding_dimension": 768,
            "model_type": "sentence-transformer",
            "training_data": "SciBERT fine-tuned on SNLI, MultiNLI, and STS-B",
            "strengths": [
                "Sentence-transformer variant of SciBERT",
                "Improved semantic similarity for scientific text",
                "Balances biomedical and scientific coverage",
            ],
            "use_cases": [
                "Scientific sentence similarity",
                "Biomedical literature matching",
                "Research paraphrase detection",
            ],
            "loading_method": "sentence-transformers",
            "recommended": True,
            "priority": 2,
        },
        "pritamdeka/S-Bluebert-snli-multinli-stsb": {
            "organization": "pritamdeka",
            "base_model": "BlueBERT",
            "size_category": "small",
            "parameters": "110M",
            "memory_gb": 0.4,
            "embedding_dimension": 768,
            "model_type": "sentence-transformer",
            "training_data": "BlueBERT fine-tuned on SNLI, MultiNLI, and STS-B",
            "strengths": [
                "Sentence-transformer optimized BlueBERT",
                "Handles both clinical and biomedical literature",
                "Good balance of efficiency and quality",
            ],
            "use_cases": [
                "Clinical sentence similarity",
                "Cross-domain medical retrieval",
                "Terminology alignment across sources",
            ],
            "loading_method": "sentence-transformers",
            "recommended": True,
            "priority": 2,
        },
        "pritamdeka/S-Biomed-Roberta-snli-multinli-stsb": {
            "organization": "pritamdeka",
            "base_model": "BioMed-RoBERTa",
            "size_category": "small",
            "parameters": "125M",
            "memory_gb": 0.5,
            "embedding_dimension": 768,
            "model_type": "sentence-transformer",
            "training_data": "BioMed-RoBERTa fine-tuned on SNLI, MultiNLI, and STS-B",
            "strengths": [
                "RoBERTa-based biomedical sentence transformer",
                "Strong semantic similarity performance",
                "Robust to varied scientific phrasing",
            ],
            "use_cases": [
                "Biomedical semantic similarity",
                "Scientific fact retrieval",
                "Medical knowledge base construction",
            ],
            "loading_method": "sentence-transformers",
            "recommended": True,
            "priority": 2,
        },
        "NeuML/pubmedbert-base-embeddings-matryoshka": {
            "organization": "NeuML",
            "base_model": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
            "size_category": "small",
            "parameters": "109.5M",
            "memory_gb": 0.4,
            "embedding_dimension": 768,
            "model_type": "sentence-transformer",
            "training_data": "PubMed abstracts and full-text articles",
            "strengths": [
                "Matryoshka representation learning for variable embedding sizes",
                "Flexible dimensionality (768 down to 64)",
                "Maintains medical retrieval performance across dimensions",
            ],
            "use_cases": [
                "Resource-constrained medical deployments",
                "Multi-resolution medical search",
                "Clinical applications needing adjustable embedding size",
            ],
            "loading_method": "sentence-transformers",
            "recommended": True,
            "priority": 1,
        },
        "pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb": {
            "organization": "pritamdeka",
            "base_model": "PubMedBERT",
            "size_category": "small",
            "parameters": "110M",
            "memory_gb": 0.4,
            "embedding_dimension": 768,
            "model_type": "sentence-transformer",
            "training_data": "PubMedBERT fine-tuned on MNLI, SNLI, SciNLI, SciTail, MedNLI, and STS-B",
            "strengths": [
                "Medical sentence embeddings built on PubMedBERT",
                "Combines biomedical literature and inference supervision",
                "Strong at medical concept similarity",
            ],
            "use_cases": [
                "Medical semantic similarity",
                "Clinical question answering",
                "Biomedical document clustering",
            ],
            "loading_method": "sentence-transformers",
            "recommended": True,
            "priority": 2,
        },
        "pritamdeka/S-PubMedBert-MS-MARCO-SCIFACT": {
            "organization": "pritamdeka",
            "base_model": "PubMedBERT",
            "size_category": "small",
            "parameters": "110M",
            "memory_gb": 0.4,
            "embedding_dimension": 768,
            "model_type": "sentence-transformer",
            "training_data": "PubMedBERT fine-tuned on MS MARCO and SciFact",
            "strengths": [
                "Combines biomedical retrieval and scientific fact verification",
                "Optimized for medical information retrieval",
                "Maintains strong semantic alignment",
            ],
            "use_cases": [
                "Biomedical information retrieval",
                "Scientific fact checking",
                "Medical literature search",
            ],
            "loading_method": "sentence-transformers",
            "recommended": True,
            "priority": 2,
        },
        "sentence-transformers/all-MiniLM-L6-v2": {
            "organization": "sentence-transformers",
            "base_model": "MiniLM",
            "size_category": "tiny",
            "parameters": "22.7M",
            "memory_gb": 0.1,
            "embedding_dimension": 384,
            "model_type": "sentence-transformer",
            "training_data": "Diverse multi-domain corpus including scientific literature",
            "strengths": [
                "Extremely fast and lightweight",
                "Surprisingly strong performance on medical text",
                "Ideal baseline for resource-constrained settings",
            ],
            "use_cases": [
                "Baseline medical embedding",
                "High-throughput clinical text processing",
                "Edge deployments and rapid prototyping",
            ],
            "loading_method": "sentence-transformers",
            "recommended": True,
            "priority": 3,
        },
        "sentence-transformers/all-MiniLM-L12-v2": {
            "organization": "sentence-transformers",
            "base_model": "MiniLM",
            "size_category": "tiny",
            "parameters": "33.4M",
            "memory_gb": 0.15,
            "embedding_dimension": 384,
            "model_type": "sentence-transformer",
            "training_data": "Diverse multi-domain corpus including scientific literature",
            "strengths": [
                "Improved quality over L6 while remaining efficient",
                "Good general-purpose medical applicability",
                "Fast inference with minimal resources",
            ],
            "use_cases": [
                "Clinical similarity at scale",
                "Medical chatbot retrieval",
                "Prototype biomedical search systems",
            ],
            "loading_method": "sentence-transformers",
            "recommended": True,
            "priority": 3,
        },
        "sentence-transformers/all-mpnet-base-v2": {
            "organization": "sentence-transformers",
            "base_model": "MPNet",
            "size_category": "small",
            "parameters": "109.5M",
            "memory_gb": 0.4,
            "embedding_dimension": 768,
            "model_type": "sentence-transformer",
            "training_data": "Large-scale sentence embedding corpus including scientific domains",
            "strengths": [
                "High-quality general-purpose embeddings",
                "Strong baseline for medical semantic similarity",
                "Well-tested and widely available",
            ],
            "use_cases": [
                "Medical document clustering",
                "Clinical question answering",
                "Biomedical semantic search",
            ],
            "loading_method": "sentence-transformers",
            "recommended": True,
            "priority": 3,
        },
        "sentence-transformers/multi-qa-mpnet-base-dot-v1": {
            "organization": "sentence-transformers",
            "base_model": "MPNet",
            "size_category": "small",
            "parameters": "109.5M",
            "memory_gb": 0.4,
            "embedding_dimension": 768,
            "model_type": "sentence-transformer",
            "training_data": "Multi-dataset QA corpus optimized for dot-product retrieval",
            "strengths": [
                "Excellent for question-answer retrieval",
                "Performs well on medical QA benchmarks",
                "Optimized for dot-product similarity",
            ],
            "use_cases": [
                "Medical FAQ systems",
                "Clinical question retrieval",
                "Healthcare knowledge base search",
            ],
            "loading_method": "sentence-transformers",
            "recommended": True,
            "priority": 3,
        },
        "sentence-transformers/msmarco-distilbert-base-v4": {
            "organization": "sentence-transformers",
            "base_model": "DistilBERT",
            "size_category": "tiny",
            "parameters": "66.4M",
            "memory_gb": 0.25,
            "embedding_dimension": 768,
            "model_type": "sentence-transformer",
            "training_data": "MS MARCO passage ranking dataset",
            "strengths": [
                "Efficient information retrieval model",
                "Balances speed and quality",
                "Performs well for medical semantic search",
            ],
            "use_cases": [
                "Medical information retrieval",
                "Clinical knowledge base search",
                "High-volume medical document indexing",
            ],
            "loading_method": "sentence-transformers",
            "recommended": True,
            "priority": 3,
        },
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": {
            "organization": "sentence-transformers",
            "base_model": "MiniLM",
            "size_category": "small",
            "parameters": "117.7M",
            "memory_gb": 0.45,
            "embedding_dimension": 384,
            "model_type": "sentence-transformer",
            "training_data": "Multilingual paraphrase corpus covering 50+ languages",
            "strengths": [
                "Multilingual coverage for international medical text",
                "Efficient embeddings for cross-lingual tasks",
                "Handles medical terminology in many languages",
            ],
            "use_cases": [
                "Multilingual medical terminology search",
                "Cross-lingual clinical applications",
                "Global healthcare chatbot systems",
            ],
            "loading_method": "sentence-transformers",
            "recommended": True,
            "priority": 3,
        },
    }

    return medical_models


def print_medical_models_guide():
    """
    Print a comprehensive guide of available medical/clinical embedding models.
    """
    print("🏥 Medical & Clinical Embedding Models Guide")
    print("=" * 80)
    print(
        "All models have ≤8B parameters and are optimized for medical/clinical text.\n"
    )

    models = get_medical_models()

    # Group by priority
    priority_groups = {}
    for model_name, info in models.items():
        priority = info.get("priority", 99)
        if priority not in priority_groups:
            priority_groups[priority] = []
        priority_groups[priority].append((model_name, info))

    for priority in sorted(priority_groups.keys()):
        print(f"\n{'='*80}")
        if priority == 1:
            print("⭐ HIGHEST PRIORITY - Small Language Models (SLMs)")
            print("   Smallest models with excellent performance - try these first!")
        elif priority == 2:
            print("⭐ RECOMMENDED - Efficient Medical Models")
            print("   Well-established models with proven performance")
        elif priority == 3:
            print("⭐ ADVANCED - Larger Models")
            print("   More parameters, potentially better for complex tasks")
        print("=" * 80)

        for model_name, info in priority_groups[priority]:
            print(f"\n📊 {model_name}")
            print(f"   Organization: {info['organization']}")
            print(f"   Base: {info['base_model']}")
            print(
                f"   Size: {info['parameters']} parameters | {info['memory_gb']} GB memory | {info['embedding_dimension']} dimensions"
            )
            print(f"   Type: {info['model_type']}")
            print(f"   Training: {info['training_data']}")

            print("   ✓ Strengths:")
            for strength in info["strengths"]:
                print(f"     • {strength}")

            print("   🎯 Use Cases:")
            for use_case in info["use_cases"]:
                print(f"     • {use_case}")

            print(f"   📦 Load with: {info['loading_method']}")

            if "benchmark_performance" in info:
                print("   📈 Benchmarks:")
                for bench, score in info["benchmark_performance"].items():
                    print(f"     • {bench}: {score}")

    print("\n" + "=" * 80)
    print("💡 RECOMMENDATIONS:")
    print("   1. Start with NeuML/pubmedbert-base-embeddings (highest priority)")
    print("   2. For clinical notes: emilyalsentzer/Bio_ClinicalBERT")
    print("   3. For long documents: lokeshch19/ModernPubMedBERT (2048 tokens)")
    print("   4. For IR tasks: pritamdeka/S-PubMedBert-MS-MARCO")
    print("=" * 80)


def get_recommended_medical_models(
    task_type: str = "general", max_memory_gb: float = 2.0
) -> List[str]:
    """
    Get recommended medical models filtered by task type and memory constraints.

    Args:
        task_type: Type of task ('general', 'clinical_notes', 'literature', 'ir', 'long_context')
        max_memory_gb: Maximum memory usage in GB

    Returns:
        List[str]: List of recommended model names
    """
    models = get_medical_models()

    # Filter by memory
    filtered = {
        name: info
        for name, info in models.items()
        if info["memory_gb"] <= max_memory_gb
    }

    # Task-specific filtering
    task_keywords = {
        "general": ["biomedical", "medical", "clinical"],
        "clinical_notes": ["clinical", "EHR", "MIMIC"],
        "literature": ["PubMed", "medical literature"],
        "ir": ["information retrieval", "search"],
        "long_context": ["long", "2048"],
    }

    if task_type != "general":
        keywords = task_keywords.get(task_type, [])
        filtered = {
            name: info
            for name, info in filtered.items()
            if any(
                kw.lower() in " ".join(info["use_cases"] + info["strengths"]).lower()
                for kw in keywords
            )
        }

    # Sort by priority
    sorted_models = sorted(
        filtered.items(), key=lambda x: (x[1]["priority"], x[1]["memory_gb"])
    )

    return [name for name, _ in sorted_models]
