import importlib.util

import psutil

# probed, not imported; get_embeddings imports the class itself
SENTENCE_TRANSFORMERS_AVAILABLE = (
    importlib.util.find_spec("sentence_transformers") is not None
)


def setup_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "Matplotlib is not installed. Install it with: pip install matplotlib"
        ) from None

    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial"]
    plt.rcParams["axes.labelweight"] = "normal"

    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = "Arial"
    plt.rcParams["mathtext.it"] = "Arial:italic"
    plt.rcParams["mathtext.bf"] = "Arial:bold"


def get_system_usage() -> dict[str, float]:
    """Current CPU and RAM usage, as percentages."""
    return {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "ram_percent": psutil.virtual_memory().percent,
    }


def get_gpu_usage() -> dict[str, str | float]:
    """Current GPU usage, or ``None`` when no GPU is present."""
    try:
        import torch
    except ImportError:
        return {"available": False, "error": "PyTorch not installed"}

    if not torch.cuda.is_available():
        return {"available": False}

    gpu_device = torch.cuda.current_device()
    return {
        "available": True,
        "device": gpu_device,
        "name": torch.cuda.get_device_name(gpu_device),
        "memory_used_gb": torch.cuda.memory_allocated(gpu_device) / (1024**3),
        "memory_total_gb": torch.cuda.get_device_properties(gpu_device).total_memory
        / (1024**3),
    }


def print_resource_usage(prefix: str = ""):
    """Log current system and GPU usage, tagged with ``prefix``."""
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
