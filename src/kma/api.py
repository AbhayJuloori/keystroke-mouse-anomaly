"""FastAPI scoring service backed by artifacts from the training pipeline."""

from __future__ import annotations

import os
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .metrics import population_stability_index
from .mouse import (
    ACTION_FEATURES,
    aggregate_session,
    extract_action_features,
    session_deviation_features,
)
from .risk import reason_codes


class KeystrokeRequest(BaseModel):
    user_id: str
    features: dict[str, float]


class MouseEvent(BaseModel):
    record_timestamp: float
    client_timestamp: float = 0.0
    button: str = "NoButton"
    state: str = "Move"
    x: float
    y: float


class SessionRequest(BaseModel):
    user_id: str
    events: list[MouseEvent] = Field(min_length=1)
    synthetic_emulator: bool | None = None
    synthetic_rooted_device: bool | None = None
    synthetic_vpn: bool | None = None


def _load_artifacts(artifact_dir: Path) -> dict[str, Any] | None:
    paths = {
        "keystroke": artifact_dir / "keystroke_models.joblib",
        "mouse": artifact_dir / "mouse_models.joblib",
        "risk": artifact_dir / "risk_model.joblib",
        "metadata": artifact_dir / "api_metadata.joblib",
    }
    if not all(path.exists() for path in paths.values()):
        return None
    return {name: joblib.load(path) for name, path in paths.items()}


def create_app(artifact_dir: str | Path | None = None) -> FastAPI:
    directory = Path(
        artifact_dir or os.environ.get("KMA_ARTIFACT_DIR", "artifacts")
    )
    loaded = _load_artifacts(directory)
    metadata = loaded["metadata"] if loaded is not None else {}
    drift_config = metadata.get("drift_config", {})
    channel_config = drift_config.get("channels", {})
    channels = metadata.get("reference_channels", {})
    psi_threshold = float(drift_config.get("psi_threshold", 0.2))
    psi_bins = int(drift_config.get("bins", 10))
    psi_smoothing = float(drift_config.get("smoothing", 0.5))
    recent: dict[str, deque[float]] = {}

    def window_settings(user_id: str) -> dict[str, int]:
        settings = channel_config.get(channels.get(user_id, "keystroke"), {})
        return {
            "window_size": int(settings.get("window_size", 200)),
            "minimum_window": int(settings.get("minimum_window", 50)),
        }

    def buffer(user_id: str) -> deque[float]:
        if user_id not in recent:
            recent[user_id] = deque(maxlen=window_settings(user_id)["window_size"])
        return recent[user_id]

    app = FastAPI(title="Keystroke and Mouse Anomaly Detection", version="0.1.0")

    def require_artifacts() -> dict[str, Any]:
        if loaded is None:
            raise HTTPException(status_code=503, detail="Training artifacts are unavailable")
        return loaded

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok" if loaded is not None else "not_ready", "artifacts": loaded is not None}

    @app.post("/score/keystroke")
    def score_keystroke(request: KeystrokeRequest) -> dict[str, Any]:
        artifacts = require_artifacts()
        bundle = artifacts["keystroke"]
        if request.user_id not in bundle["models"]:
            raise HTTPException(status_code=404, detail="Unknown keystroke user")
        missing = [name for name in bundle["feature_names"] if name not in request.features]
        extra = [name for name in request.features if name not in bundle["feature_names"]]
        if missing or extra:
            raise HTTPException(
                status_code=422, detail={"missing_features": missing, "extra_features": extra}
            )
        model = bundle["models"][request.user_id]
        values = np.asarray([[request.features[name] for name in bundle["feature_names"]]])
        score = float(np.mean(np.abs(values[0] - model["center"]) / model["scale"]))
        buffer(request.user_id).append(score)
        return {
            "user_id": request.user_id,
            "anomaly_score": score,
            "threshold": float(model["threshold"]),
            "decision": "anomaly" if score > model["threshold"] else "accept",
        }

    @app.post("/score/session")
    def score_session(request: SessionRequest) -> dict[str, Any]:
        artifacts = require_artifacts()
        mouse_bundle = artifacts["mouse"]
        if request.user_id not in mouse_bundle["models"]:
            raise HTTPException(status_code=404, detail="Unknown mouse user")
        rows = [event.model_dump() for event in request.events]
        events = pd.DataFrame(rows).rename(
            columns={
                "record_timestamp": "record timestamp",
                "client_timestamp": "client timestamp",
            }
        )
        actions = extract_action_features(events)
        if actions.empty:
            behavioral_anomaly = 0.5
        else:
            classifier = mouse_bundle["models"][request.user_id]
            genuine_probability = classifier.predict_proba(actions[ACTION_FEATURES])[:, 1]
            behavioral_anomaly = 1.0 - float(np.mean(genuine_probability))
            buffer(request.user_id).extend((1.0 - genuine_probability).tolist())
        deviations = session_deviation_features(
            aggregate_session(actions),
            mouse_bundle["deviation_profiles"][request.user_id],
        )
        features = pd.DataFrame(
            [
                {
                    "behavioral_anomaly": behavioral_anomaly,
                    **deviations,
                    "synthetic_emulator": int(bool(request.synthetic_emulator)),
                    "synthetic_rooted_device": int(bool(request.synthetic_rooted_device)),
                    "synthetic_vpn": int(bool(request.synthetic_vpn)),
                }
            ]
        )
        risk_bundle = artifacts["risk"]
        risk_model = risk_bundle["model"]
        risk_probability = float(risk_model.predict_proba(features)[0, 1])
        shap_row = risk_model.shap_values(features)[0]
        reasons = reason_codes(shap_row, features.iloc[0], risk_model.feature_names)
        return {
            "user_id": request.user_id,
            "risk_score": risk_probability * 100.0,
            "behavioral_anomaly_score": behavioral_anomaly,
            "deviation_features": deviations,
            "threshold": float(risk_bundle["threshold"] * 100.0),
            "decision": (
                "review" if risk_probability > risk_bundle["threshold"] else "accept"
            ),
            "top_reason_codes": reasons,
            "action_count": int(len(actions)),
            "synthetic_device_signals": True,
        }

    @app.get("/drift/{user_id}")
    def drift(user_id: str) -> dict[str, Any]:
        artifacts = require_artifacts()
        references = artifacts["metadata"]["reference_scores"]
        if user_id not in references:
            raise HTTPException(status_code=404, detail="Unknown user")
        values = np.asarray(recent.get(user_id, ()), dtype=float)
        settings = window_settings(user_id)
        psi = population_stability_index(
            references[user_id], values, bins=psi_bins, smoothing=psi_smoothing
        )
        enough = bool(values.size >= settings["minimum_window"])
        return {
            "user_id": user_id,
            "channel": channels.get(user_id, "keystroke"),
            "psi": psi,
            "recent_score_count": int(values.size),
            "minimum_window": settings["minimum_window"],
            "window_size": settings["window_size"],
            "enough_data": enough,
            "drift": bool(enough and psi > psi_threshold),
            "threshold": psi_threshold,
            "reference_source": artifacts["metadata"].get(
                "reference_sources", {}
            ).get(user_id, "unspecified"),
        }

    return app


app = create_app()
