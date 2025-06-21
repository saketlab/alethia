import importlib.resources
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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
            mteb_df = pd.read_csv(data_path)

        mteb_df = filter_huggingface_only(mteb_df)

        def extract_model_name(model_str):
            if pd.isna(model_str):
                return model_str
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

    print(f"\n🎯 RECOMMENDATIONS BY USE CASE")
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
