"""CPU thread and batch-size hints for loading and running embedding models."""

import contextlib
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _cpu_count() -> int:
    try:
        return os.cpu_count() or 1
    except Exception:
        return 1


@dataclass
class ClinicalCPUOptimizationConfig:
    """Configuration for CPU-optimized model loading and inference.

    Attributes:
        num_threads: Threads to use for CPU inference. ``None`` => auto (all cores).
        batch_size: Preferred encode batch size.
        device: Target device; CPU optimization only applies when ``"cpu"``.
        enable: Master switch; when ``False`` optimization is skipped entirely.
    """

    num_threads: Optional[int] = None
    batch_size: int = 64
    device: str = "cpu"
    enable: bool = True

    def resolved_threads(self) -> int:
        return self.num_threads or _cpu_count()


@contextlib.contextmanager
def cpu_thread_pool(num_threads: Optional[int] = None):
    """Temporarily set the torch CPU thread count for the enclosed block.

    No-op when torch is unavailable or ``num_threads`` is ``None``. The previous
    thread count is restored on exit.
    """
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


def get_cpu_runtime_hints(model_obj: Any = None) -> Dict[str, Any]:
    """Return CPU inference hints (``batch_size``, ``num_threads``) for a model.

    If ``model_obj`` carries a ``cpu_optimization_config`` mapping (as produced by the
    CPU-optimized loader), its ``batch_size`` / ``num_threads`` are honoured. Otherwise
    conservative defaults are returned (``batch_size=64``, ``num_threads=None`` meaning
    "let the backend decide").
    """
    defaults = {"batch_size": 64, "num_threads": None}
    config = getattr(model_obj, "cpu_optimization_config", None)
    if isinstance(config, dict):
        return {
            "batch_size": config.get("batch_size", defaults["batch_size"]),
            "num_threads": config.get("num_threads", defaults["num_threads"]),
        }
    return defaults


def get_cpu_optimized_model(
    model_name: str,
    config: Optional[ClinicalCPUOptimizationConfig] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> Any:
    """Load a SentenceTransformer pinned to CPU with thread tuning applied.

    Returns the loaded model, or ``None`` if optimization is disabled or
    sentence-transformers is unavailable, signalling the caller to fall back to a
    standard load path.
    """
    config = config or ClinicalCPUOptimizationConfig()
    if not config.enable:
        return None

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None

    try:
        import torch

        torch.set_num_threads(config.resolved_threads())
    except ImportError:
        pass

    try:
        model = SentenceTransformer(
            model_name,
            device="cpu",
            **(model_kwargs or {}),
        )
        logger.debug("Loaded CPU-optimized model %s", model_name)
        return model
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("CPU-optimized load failed for %s: %s", model_name, e)
        return None
