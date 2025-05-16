from typing import Union, List, Dict, Tuple
import psutil
import torch


def setup_matplotlib():
    import matplotlib.pyplot as plt

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
    if gpu_info["available"]:
        print(f"GPU: {gpu_info['name']}")
        print(
            f"{label}GPU Memory Usage: {gpu_info['memory_used_gb']:.2f}GB / {gpu_info['memory_total_gb']:.2f}GB"
        )
    else:
        if not prefix:
            print("No GPU detected, running on CPU only")

    return sys_usage, gpu_info
