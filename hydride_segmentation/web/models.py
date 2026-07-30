"""Model catalog and warm-loading for the intranet web application.

The browser app must answer the first user request quickly, so trained
checkpoints are loaded into the process-local bundle cache at startup rather
than on the first upload. All model metadata comes from the same registry the
desktop GUI and the CLI use, so the three surfaces always offer the same models.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from hydride_segmentation.microseg_adapter import get_gui_model_specs, resolve_gui_model_reference
from src.microseg.inference import warm_load_reference_bundle

_LOGGER = logging.getLogger(__name__)

#: Model identifier of the classical pipeline, which needs no checkpoint.
CONVENTIONAL_MODEL_ID = "hydride_conventional"


@dataclass
class WebModelOption:
    """One model offered in the browser model selector."""

    model_id: str
    display_name: str
    description: str
    details: str
    family: str
    available: bool
    availability: str
    availability_message: str
    is_conventional: bool
    warm_state: str = "cold"
    warm_message: str = ""
    warm_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view of this option."""

        return {
            "model_id": self.model_id,
            "display_name": self.display_name,
            "description": self.description,
            "details": self.details,
            "family": self.family,
            "available": self.available,
            "availability": self.availability,
            "availability_message": self.availability_message,
            "is_conventional": self.is_conventional,
            "warm_state": self.warm_state,
            "warm_message": self.warm_message,
            "warm_seconds": round(float(self.warm_seconds), 3),
        }


class ModelCatalog:
    """Thread-safe catalog of selectable models with startup warm-loading."""

    def __init__(
        self,
        *,
        enable_gpu: bool = False,
        device_policy: str = "cpu",
        preload_model_ids: tuple[str, ...] = (),
    ) -> None:
        self._lock = threading.Lock()
        self._enable_gpu = bool(enable_gpu)
        self._device_policy = str(device_policy)
        self._preload_model_ids = tuple(preload_model_ids)
        self._options: list[WebModelOption] = []
        self._warm_state: dict[str, dict[str, Any]] = {}
        self._preload_started = False
        self._preload_finished = False
        self._preload_seconds = 0.0
        self.refresh()

    # -- catalog ---------------------------------------------------------

    def refresh(self) -> list[WebModelOption]:
        """Reload model metadata from the registry.

        Returns
        -------
        list of WebModelOption
            The refreshed selector options, conventional pipeline included.
        """

        options: list[WebModelOption] = []
        try:
            specs = get_gui_model_specs()
        except Exception as exc:  # pragma: no cover - registry failure is environmental
            _LOGGER.warning("Could not read the model registry: %s", exc)
            specs = []

        for spec in specs:
            model_id = str(spec.get("model_id", "")).strip()
            if not model_id:
                continue
            availability = str(spec.get("availability", "ready"))
            options.append(
                WebModelOption(
                    model_id=model_id,
                    display_name=str(spec.get("display_name", model_id)),
                    description=str(spec.get("description", "")),
                    details=str(spec.get("details", "")),
                    family=str(spec.get("feature_family", "")),
                    available=availability in {"ready", "no_checkpoint_required"},
                    availability=availability,
                    availability_message=str(spec.get("availability_message", "")),
                    is_conventional=model_id == CONVENTIONAL_MODEL_ID,
                )
            )

        with self._lock:
            for option in options:
                state = self._warm_state.get(option.model_id)
                if state:
                    option.warm_state = str(state.get("state", "cold"))
                    option.warm_message = str(state.get("message", ""))
                    option.warm_seconds = float(state.get("seconds", 0.0))
            self._options = options
        return list(options)

    def options(self) -> list[WebModelOption]:
        """Return the current selector options."""

        with self._lock:
            return list(self._options)

    def get(self, model_id: str) -> WebModelOption | None:
        """Return one option by identifier, or ``None`` when it is unknown."""

        target = str(model_id).strip()
        for option in self.options():
            if option.model_id == target:
                return option
        return None

    def default_model_id(self, configured: str = "auto") -> str:
        """Resolve the model preselected in the browser.

        Parameters
        ----------
        configured:
            Configured default. ``"auto"`` picks the first available trained
            model and falls back to the conventional pipeline.

        Returns
        -------
        str
            A model identifier that is currently runnable.
        """

        options = self.options()
        wanted = str(configured).strip()
        if wanted and wanted != "auto":
            match = self.get(wanted)
            if match is not None and match.available:
                return match.model_id
            _LOGGER.warning(
                "Configured default model %r is not available; falling back to automatic selection",
                wanted,
            )

        for option in options:
            if option.available and not option.is_conventional:
                return option.model_id
        for option in options:
            if option.available:
                return option.model_id
        return CONVENTIONAL_MODEL_ID

    # -- warm loading ----------------------------------------------------

    def _preload_targets(self) -> list[WebModelOption]:
        targets = [
            option
            for option in self.options()
            if option.available and not option.is_conventional
        ]
        if self._preload_model_ids:
            allowed = set(self._preload_model_ids)
            targets = [option for option in targets if option.model_id in allowed]
        return targets

    def _set_warm_state(self, model_id: str, state: str, message: str, seconds: float = 0.0) -> None:
        with self._lock:
            self._warm_state[model_id] = {"state": state, "message": message, "seconds": float(seconds)}
            for option in self._options:
                if option.model_id == model_id:
                    option.warm_state = state
                    option.warm_message = message
                    option.warm_seconds = float(seconds)

    def warm_model(self, model_id: str) -> dict[str, Any]:
        """Load one trained model into the shared bundle cache.

        Parameters
        ----------
        model_id:
            Registry identifier of a trained model.

        Returns
        -------
        dict
            State mapping with ``state``, ``message`` and ``seconds`` keys.
        """

        option = self.get(model_id)
        if option is None:
            self._set_warm_state(model_id, "error", "Model is not registered.")
            return {"state": "error", "message": "Model is not registered.", "seconds": 0.0}
        if option.is_conventional:
            self._set_warm_state(model_id, "ready", "Classical pipeline; nothing to load.")
            return {"state": "ready", "message": "Classical pipeline; nothing to load.", "seconds": 0.0}
        if not option.available:
            self._set_warm_state(model_id, "unavailable", option.availability_message)
            return {"state": "unavailable", "message": option.availability_message, "seconds": 0.0}

        self._set_warm_state(model_id, "loading", "Loading model into memory...")
        started = time.perf_counter()
        try:
            reference = resolve_gui_model_reference(option.display_name, {})
            if reference is None:
                raise RuntimeError("model reference could not be resolved from the registry")
            status = warm_load_reference_bundle(
                reference,
                enable_gpu=self._enable_gpu,
                device_policy=self._device_policy,
            )
        except Exception as exc:
            elapsed = max(0.0, time.perf_counter() - started)
            message = f"Model could not be loaded: {exc}"
            _LOGGER.warning("Warm load failed for %s: %s", model_id, exc)
            self._set_warm_state(model_id, "error", message, elapsed)
            return {"state": "error", "message": message, "seconds": elapsed}

        elapsed = max(0.0, time.perf_counter() - started)
        self._set_warm_state(model_id, "ready", status.message, elapsed)
        _LOGGER.info("Warm load complete for %s in %.2fs (%s)", model_id, elapsed, status.message)
        return {"state": "ready", "message": status.message, "seconds": elapsed}

    def preload(self) -> dict[str, Any]:
        """Warm every preload target and return a summary.

        Returns
        -------
        dict
            Summary with per-model results and the total elapsed time.
        """

        with self._lock:
            self._preload_started = True
            self._preload_finished = False
        started = time.perf_counter()
        results: dict[str, Any] = {}
        targets = self._preload_targets()
        if not targets:
            _LOGGER.info(
                "No trained model is available to preload; the classical pipeline is ready immediately"
            )
        for option in targets:
            results[option.model_id] = self.warm_model(option.model_id)
        elapsed = max(0.0, time.perf_counter() - started)
        with self._lock:
            self._preload_finished = True
            self._preload_seconds = elapsed
        _LOGGER.info("Model preload finished in %.2fs for %d model(s)", elapsed, len(targets))
        return {"models": results, "seconds": elapsed, "count": len(targets)}

    def preload_async(self) -> threading.Thread:
        """Start :meth:`preload` on a daemon thread and return it."""

        thread = threading.Thread(target=self.preload, name="microseg-web-preload", daemon=True)
        thread.start()
        return thread

    def status(self) -> dict[str, Any]:
        """Return readiness information for the status endpoint."""

        options = self.options()
        with self._lock:
            preload_started = self._preload_started
            preload_finished = self._preload_finished
            preload_seconds = self._preload_seconds
        trained = [option for option in options if not option.is_conventional]
        return {
            "preload_started": preload_started,
            "preload_finished": preload_finished,
            "preload_seconds": round(float(preload_seconds), 3),
            "trained_model_count": len(trained),
            "ready_model_count": sum(1 for option in options if option.warm_state == "ready"),
            "conventional_available": any(option.is_conventional and option.available for option in options),
            "models": [option.to_dict() for option in options],
        }
