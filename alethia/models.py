import importlib.resources
import re

import pandas as pd


def filter_huggingface_only(mteb_df: pd.DataFrame) -> pd.DataFrame:
    """Filter MTEB DataFrame to keep only HuggingFace models."""
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
            if "/" in clean_name and not clean_name.startswith("models/"):
                if not any(
                    api in clean_name.lower()
                    for api in ["gpt-", "text-embedding", "gemini-", "claude-"]
                ):
                    return True
        return False

    return mteb_df[mteb_df["Model"].apply(is_huggingface_model)].copy()


def load_mteb_dashboard_data() -> pd.DataFrame:
    """Load and process MTEB dashboard data from package resources."""
    try:
        with importlib.resources.path(
            "alethia.data", "mteb_dashboard.csv"
        ) as data_path:
            mteb_df = pd.read_csv(data_path)

        mteb_df = filter_huggingface_only(mteb_df)

        def extract_model_name(model_str):
            if pd.isna(model_str):
                return model_str
            match = re.search(r"\[(.*?)\]\((.*?)\)", str(model_str))
            if match:
                url = match.group(2)
                hf_match = re.search(r"huggingface\.co/([^/?]+/[^/?]+)", url)
                if hf_match:
                    return hf_match.group(1)
                return match.group(1)
            match = re.search(r"\[(.*?)\]", str(model_str))
            return match.group(1) if match else str(model_str)

        from .utils import convert_memory_to_gb

        mteb_df["clean_model_name"] = mteb_df["Model"].apply(extract_model_name)
        mteb_df["memory_gb"] = mteb_df["Memory Usage (MB)"].apply(convert_memory_to_gb)
        mteb_df["clean_parameters"] = mteb_df["Number of Parameters"].fillna("Unknown")

        return mteb_df

    except Exception as e:
        print(f"Warning: Could not load MTEB data: {e}")
        return pd.DataFrame()
