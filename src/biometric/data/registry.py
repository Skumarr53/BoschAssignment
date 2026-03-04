"""Registry pattern for datasets and transforms.

Enables adding new modalities without modifying core code.
Thread-safe via a module-level lock. Built-in registrations happen
in register_builtins() which is called lazily on first lookup.
"""

import threading
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")

_DATASET_REGISTRY: dict[str, tuple[type, dict[str, Any]]] = {}
_TRANSFORM_REGISTRY: dict[str, tuple[Callable[..., Any], dict[str, Any]]] = {}
_lock = threading.Lock()
_builtins_registered = False


def _ensure_builtins() -> None:
    """Register built-in datasets and transforms on first use."""
    global _builtins_registered
    if _builtins_registered:
        return
    with _lock:
        if _builtins_registered:
            return
        from biometric.data.dataset import (
            FingerprintDataset,
            IrisDataset,
            MultimodalBiometricDataset,
        )
        from biometric.data.preprocessing import (
            get_fingerprint_transform,
            get_iris_transform,
        )

        _DATASET_REGISTRY["fingerprint"] = (FingerprintDataset, {})
        _DATASET_REGISTRY["iris"] = (IrisDataset, {})
        _DATASET_REGISTRY["multimodal"] = (MultimodalBiometricDataset, {})
        _TRANSFORM_REGISTRY["fingerprint"] = (
            get_fingerprint_transform,
            {"size": (96, 96), "train": False},
        )
        _TRANSFORM_REGISTRY["iris"] = (get_iris_transform, {"size": (224, 224), "train": False})
        _builtins_registered = True


def register_dataset(
    name: str, default_kwargs: dict[str, Any] | None = None
) -> Callable[[type[T]], type[T]]:
    """Decorator to register a Dataset class."""

    def _register(cls: type[T]) -> type[T]:
        with _lock:
            _DATASET_REGISTRY[name] = (cls, default_kwargs or {})
        return cls

    return _register


def register_transform(
    modality: str,
    default_kwargs: dict[str, Any] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to register a transform factory."""

    def _register(fn: Callable[..., Any]) -> Callable[..., Any]:
        with _lock:
            _TRANSFORM_REGISTRY[modality] = (fn, default_kwargs or {})
        return fn

    return _register


def get_dataset(name: str, **kwargs: Any) -> Any:
    """Get a registered dataset by name with merged kwargs."""
    _ensure_builtins()
    if name not in _DATASET_REGISTRY:
        msg = f"Unknown dataset: {name}. Registered: {list(_DATASET_REGISTRY)}"
        raise KeyError(msg)
    cls, defaults = _DATASET_REGISTRY[name]
    merged = {**defaults, **kwargs}
    return cls(**merged)


def get_transform(modality: str, **kwargs: Any) -> Any:
    """Get a registered transform by modality with merged kwargs."""
    _ensure_builtins()
    if modality not in _TRANSFORM_REGISTRY:
        msg = f"Unknown modality: {modality}. Registered: {list(_TRANSFORM_REGISTRY)}"
        raise KeyError(msg)
    fn, defaults = _TRANSFORM_REGISTRY[modality]
    merged = {**defaults, **kwargs}
    return fn(**merged)


def list_datasets() -> list[str]:
    """Return registered dataset names."""
    _ensure_builtins()
    return list(_DATASET_REGISTRY)


def list_transforms() -> list[str]:
    """Return registered transform modalities."""
    _ensure_builtins()
    return list(_TRANSFORM_REGISTRY)
