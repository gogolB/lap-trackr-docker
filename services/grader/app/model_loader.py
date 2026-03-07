"""Factory that loads the active model backend based on DB state.

Caches the loaded backend so it isn't reloaded on every grading job.
"""

from __future__ import annotations

import logging
from typing import Any

from app.backends.base import ModelBackend
from app.backends.placeholder_backend import PlaceholderBackend
from app.db import get_active_model_info

logger = logging.getLogger("grader.model_loader")

# Cache: (model_id, backend_instance)
_cached: tuple[str | None, ModelBackend | None] = (None, None)

# Map model_type strings to backend classes
_BACKEND_CLASSES: dict[str, type[ModelBackend]] = {}


def _get_backend_class(model_type: str) -> type[ModelBackend]:
    """Lazy-import backend classes to avoid heavy imports at module load."""
    if not _BACKEND_CLASSES:
        from app.backends.cotracker_backend import CoTrackerBackend
        from app.backends.sam2_backend import SAM2Backend
        from app.backends.tapir_backend import TAPIRBackend
        from app.backends.yolo_backend import YOLOBackend

        _BACKEND_CLASSES.update({
            "yolo": YOLOBackend,
            "cotracker": CoTrackerBackend,
            "sam2": SAM2Backend,
            "tapir": TAPIRBackend,
        })

    cls = _BACKEND_CLASSES.get(model_type)
    if cls is None:
        raise ValueError(f"Unknown model type: {model_type}")
    return cls


def get_backend() -> ModelBackend:
    """Return the cached backend for the currently active model.

    If no model is active, falls back to the placeholder backend.
    If the active model changed since last call, unloads the old backend
    and loads the new one.
    """
    global _cached

    info = get_active_model_info()

    if info is None:
        # No active model in DB — use placeholder
        cached_id, cached_backend = _cached
        if cached_id is None and cached_backend is not None:
            return cached_backend

        logger.info("No active model found; using placeholder backend")
        backend = PlaceholderBackend()
        backend.load("")
        _cached = (None, backend)
        return backend

    model_id = str(info["id"])
    model_type = info["model_type"]
    file_path = info["file_path"]

    cached_id, cached_backend = _cached
    if cached_id == model_id and cached_backend is not None:
        return cached_backend

    # Unload previous backend
    if cached_backend is not None:
        try:
            cached_backend.unload()
        except Exception:
            logger.warning("Failed to unload previous backend", exc_info=True)

    # Load new backend
    logger.info("Loading backend: type=%s, id=%s, path=%s", model_type, model_id, file_path)
    cls = _get_backend_class(model_type)
    backend = cls()
    try:
        backend.load(file_path)
    except Exception:
        logger.exception("Failed to load backend %s (id=%s), falling back to placeholder", model_type, model_id)
        try:
            backend.unload()
        except Exception:
            logger.warning("Failed to unload partially-initialized backend", exc_info=True)
        placeholder = PlaceholderBackend()
        placeholder.load("")
        _cached = (None, placeholder)
        return placeholder
    _cached = (model_id, backend)
    return backend
