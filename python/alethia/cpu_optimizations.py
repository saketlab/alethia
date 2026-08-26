"""CPU thread and batch-size hints for loading and running embedding models."""

import contextlib
from typing import Any


@contextlib.contextmanager
def cpu_thread_pool(num_threads: int | None = None):
    """Temporarily set the torch CPU thread count, restoring it on exit."""
    if num_threads is None:
        yield
        return

    try:
        import torch
    except ImportError:
        yield
        return

    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(int(num_threads))
        yield
    finally:
        try:
            torch.set_num_threads(previous)
        except Exception:
            pass


def get_cpu_runtime_hints(model_obj: Any = None) -> dict[str, Any]:
    """CPU inference hints for a model: ``batch_size`` and ``num_threads``."""
    defaults = {"batch_size": 64, "num_threads": None}
    config = getattr(model_obj, "cpu_optimization_config", None)
    if isinstance(config, dict):
        return {
            "batch_size": config.get("batch_size", defaults["batch_size"]),
            "num_threads": config.get("num_threads", defaults["num_threads"]),
        }
    return defaults
