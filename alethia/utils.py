from typing import Dict, Union

import pandas as pd
import psutil


def convert_memory_to_gb(memory_str):
    if pd.isna(memory_str) or memory_str == "Unknown":
        return None
    try:
        return round(float(memory_str) / 1024, 2)
    except (ValueError, TypeError):
        return None


def setup_matplotlib():
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
            "axes.labelweight": "normal",
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
        }
    )


def get_system_usage() -> Dict[str, float]:
    return {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "ram_percent": psutil.virtual_memory().percent,
    }


def get_gpu_usage() -> Dict[str, Union[str, float]]:
    try:
        import torch
    except ImportError:
        return {"available": False, "error": "PyTorch not installed"}

    if not torch.cuda.is_available():
        return {"available": False}

    device = torch.cuda.current_device()
    return {
        "available": True,
        "device": device,
        "name": torch.cuda.get_device_name(device),
        "memory_used_gb": torch.cuda.memory_allocated(device) / (1024**3),
        "memory_total_gb": torch.cuda.get_device_properties(device).total_memory
        / (1024**3),
    }


def print_resource_usage(prefix: str = ""):
    label = f"{prefix} " if prefix else ""
    sys_usage = get_system_usage()
    print(f"{label}CPU Usage: {sys_usage['cpu_percent']}%")
    print(f"{label}RAM Usage: {sys_usage['ram_percent']}%")

    gpu_info = get_gpu_usage()
    if gpu_info.get("available"):
        print(f"GPU: {gpu_info['name']}")
        print(
            f"{label}GPU Memory: {gpu_info['memory_used_gb']:.2f}GB / {gpu_info['memory_total_gb']:.2f}GB"
        )
    elif not prefix:
        print(gpu_info.get("error", "No GPU detected"))

    return sys_usage, gpu_info
