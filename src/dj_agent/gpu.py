"""GPU memory manager — ensures only one heavy model is loaded at a time.

Prevents CUDA OOM by enforcing mutual exclusion between large models
(Flamingo ~15GB, Roformer ~4GB, Beat This ~78MB, CLAP ~600MB).

Usage::

    from dj_agent.gpu import gpu_manager

    with gpu_manager.acquire("flamingo"):
        model = load_heavy_model()
        result = model(audio)
    # Model should be unloaded after this block
"""

from __future__ import annotations

import gc
import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)


class GPUManager:
    """Singleton that tracks which model owns GPU memory.

    Only one heavy model can be loaded at a time.  When a new model
    requests the GPU, the previous one is evicted first.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current_owner: str | None = None
        self._unload_callbacks: dict[str, Any] = {}

    def register_unloader(self, name: str, callback: Any) -> None:
        """Register a callback that unloads a model from GPU."""
        self._unload_callbacks[name] = callback

    def acquire(self, name: str) -> "_GPUContext":
        """Context manager that ensures this model has GPU access.

        Evicts any other model first.
        """
        return _GPUContext(self, name)

    def _ensure_owner(self, name: str) -> None:
        """Evict current owner if different, then claim ownership."""
        with self._lock:
            if self._current_owner == name:
                return

            # Evict previous owner
            if self._current_owner and self._current_owner in self._unload_callbacks:
                logger.info("GPU: evicting '%s' to load '%s'", self._current_owner, name)
                try:
                    self._unload_callbacks[self._current_owner]()
                except Exception as e:
                    logger.warning("GPU: eviction of '%s' failed: %s", self._current_owner, e)

            # Force CUDA cache cleanup
            _clear_cuda_cache()

            self._current_owner = name

    def release(self, name: str) -> None:
        """Release GPU ownership (optional — next acquire will evict anyway)."""
        with self._lock:
            if self._current_owner == name:
                if name in self._unload_callbacks:
                    try:
                        self._unload_callbacks[name]()
                    except Exception:
                        pass
                _clear_cuda_cache()
                self._current_owner = None


class _GPUContext:
    def __init__(self, manager: GPUManager, name: str):
        self._manager = manager
        self._name = name

    def __enter__(self):
        self._manager._ensure_owner(self._name)
        return self

    def __exit__(self, *exc):
        # Don't auto-release — keep model cached until someone else needs GPU
        pass


def _clear_cuda_cache() -> None:
    """Clear CUDA memory cache and force garbage collection."""
    gc.collect()  # Python GC first to drop references
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # ensure all ops complete before next model loads
    except ImportError:
        pass


# Global singleton
gpu_manager = GPUManager()
