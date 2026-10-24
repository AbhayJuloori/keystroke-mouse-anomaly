"""Synthetic device signals, grouped risk evaluation, and reason codes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
import yaml
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from .metrics import equal_error_rate, global_far_operating_point
from .mouse import DEVIATION_FEATURES


BEHAVIOR_FEATURES = ["behavioral_anomaly"]
DEVICE_FEATURES = [
    "synthetic_emulator",
    "synthetic_rooted_device",
    "synthetic_vpn",
]
BEHAVIOR_DEVIATION_FEATURES = BEHAVIOR_FEATURES + DEVIATION_FEATURES
ALL_FEATURES = BEHAVIOR_DEVIATION_FEATURES + DEVICE_FEATURES
MODEL_CANDIDATES = ("logistic_regression", "gradient_boosting")


@dataclass
class RiskModel:
    model: Any
    feature_names: list[str]
    model_name: str
    scaler: StandardScaler | None = None
    _explainer: Any = field(default=None, init=False, repr=False)

    def _values(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame[self.feature_names].to_numpy(dtype=float)
        return self.scaler.transform(values) if self.scaler is not None else values

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._values(frame))

    def shap_values(self, frame: pd.DataFrame) -> np.ndarray:
        values = self._values(frame)
        if self._explainer is None:
            if self.model_name == "logistic_regression":
                self._explainer = shap.LinearExplainer(
                    self.model, np.zeros((1, len(self.feature_names)))
                )
            else:
                self._explainer = shap.TreeExplainer(self.model)
        explained = self._explainer(values)
        result = np.asarray(explained.values)
        if result.ndim == 3:
            result = result[:, :, -1]
        return result

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_explainer"] = None
        return state


# Retained as an import-compatible alias for small external fixtures.
ScaledLogisticModel = RiskModel


def add_synthetic_device_signals(
    sessions: pd.DataFrame, config_path: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate explicitly synthetic flags conditional on the benchmark label."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    rng = np.random.default_rng(int(config["seed"]))
    output = sessions.copy()
    for column in DEVICE_FEATURES:
        signal = column.removeprefix("synthetic_")
        probabilities = output["label"].map(
            {
                0: float(config["rates"]["genuine"][signal]),
                1: float(config["rates"]["impostor"][signal]),
            }
        )
        output[column] = (rng.random(len(output)) < probabilities.to_numpy()).astype(int)
    return output, config


def _fit_model(
    frame: pd.DataFrame, features: list[str], model_name: str
) -> RiskModel:
    if model_name == "logistic_regression":
        values = frame[features].to_numpy(dtype=float)
        scaler = StandardScaler().fit(values)
        model = LogisticRegression(
            class_weight="balanced", C=1.0, max_iter=1000, random_state=42
        ).fit(scaler.transform(values), frame["label"])
        return RiskModel(model, features, model_name, scaler)
    if model_name == "gradient_boosting":
        values = frame[features].to_numpy(dtype=float)
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
            random_state=42,
        ).fit(values, frame["label"])
        return RiskModel(model, features, model_name)
    raise ValueError(f"Unknown risk model: {model_name}")


def _oof_predictions(
    frame: pd.DataFrame, features: list[str], model_name: str
) -> np.ndarray:
    predictions = np.zeros(len(frame), dtype=float)
    splitter = GroupKFold(n_splits=5)
    for train_index, validation_index in splitter.split(
        frame, frame["label"], frame["user_id"]
    ):
        train = frame.iloc[train_index]
        validation = frame.iloc[validation_index]
        fitted = _fit_model(train, features, model_name)
        predictions[validation_index] = fitted.predict_proba(validation)[:, 1]
    return predictions


def _score_summary(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    genuine = scores[labels == 0]
    impostor = scores[labels == 1]
    eer, threshold, far, frr = equal_error_rate(genuine, impostor)
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "eer": eer,
        "eer_threshold": threshold,
        "eer_far": far,
        "eer_frr": frr,
    }


def reason_codes(
    shap_row: np.ndarray,
    feature_values: pd.Series,
    feature_names: list[str] | None = None,
) -> list[str]:
    """Map the three largest absolute SHAP contributions to feature-specific text."""
    names = feature_names or ALL_FEATURES
    fixed_messages = {
        "behavioral_anomaly": (
            "overall mouse behavior unlike this user's history",
            "overall mouse behavior consistent with this user's history",
        ),
        "synthetic_emulator": (
            "emulator detected (synthetic)",
            "no emulator signal (synthetic)",
        ),
        "synthetic_rooted_device": (
            "rooted device detected (synthetic)",
            "no rooted-device signal (synthetic)",
        ),
        "synthetic_vpn": (
            "VPN detected (synthetic)",
            "no VPN signal (synthetic)",
        ),
    }
    deviation_messages = {
        "deviation_speed_mean": ("mouse speed much faster", "mouse speed much slower"),
        "deviation_speed_max": ("peak mouse speed much faster", "peak mouse speed much slower"),
        "deviation_curvature_mean": ("path curvature higher", "path curvature lower"),
        "deviation_straightness": ("mouse paths straighter", "mouse paths less straight"),
        "deviation_click_duration": ("click hold time longer", "click hold time shorter"),
        "deviation_acceleration_mean": ("mouse acceleration higher", "mouse acceleration lower"),
        "deviation_angular_velocity_mean": ("turning speed higher", "turning speed lower"),
        "deviation_action_duration": ("mouse actions longer", "mouse actions shorter"),
    }
    order = np.argsort(np.abs(shap_row))[::-1][:3]
    output = []
    for index in order:
        feature = names[int(index)]
        value = float(feature_values[feature])
        if feature in deviation_messages:
            direction = deviation_messages[feature][0 if value >= 0 else 1]
            output.append(f"{direction} than this user's history")
        elif feature in DEVICE_FEATURES:
            output.append(fixed_messages[feature][0 if value >= 0.5 else 1])
        else:
            output.append(fixed_messages[feature][0 if float(shap_row[index]) >= 0 else 1])
    return output


def train_risk_model(
    sessions: pd.DataFrame, config_path: Path, artifact_dir: Path
) -> tuple[dict[str, Any], pd.DataFrame, RiskModel, np.ndarray]:
    """Compare grouped risk models, report three feature stages, and fit serving."""
    frame, config = add_synthetic_device_signals(sessions, config_path)
    labels = frame["label"].to_numpy(dtype=int)

    candidate_predictions = {
        name: _oof_predictions(frame, ALL_FEATURES, name) for name in MODEL_CANDIDATES
    }
    comparison = {
        name: _score_summary(labels, predictions)
        for name, predictions in candidate_predictions.items()
    }
    selected_name = max(
        MODEL_CANDIDATES,
        key=lambda name: (comparison[name]["auc"], name == "logistic_regression"),
    )

    stages = {
        "behavior_only": BEHAVIOR_FEATURES,
        "behavior_plus_deviations": BEHAVIOR_DEVIATION_FEATURES,
        "behavior_plus_deviations_plus_synthetic": ALL_FEATURES,
    }
    stage_predictions: dict[str, np.ndarray] = {}
    stage_results: dict[str, dict[str, float]] = {}
    for stage_name, features in stages.items():
        predictions = (
            candidate_predictions[selected_name]
            if features == ALL_FEATURES
            else _oof_predictions(frame, features, selected_name)
        )
        stage_predictions[stage_name] = predictions
        stage_results[stage_name] = _score_summary(labels, predictions)
        frame[f"risk_{stage_name}"] = predictions * 100.0

    full_oof = stage_predictions["behavior_plus_deviations_plus_synthetic"]
    full_model = _fit_model(frame, ALL_FEATURES, selected_name)
    shap_values = full_model.shap_values(frame)
    frame["reason_codes"] = [
        reason_codes(shap_values[index], frame.iloc[index], ALL_FEATURES)
        for index in range(len(frame))
    ]
    operating_points = {
        "global_far_1pct": global_far_operating_point(
            full_oof[labels == 0], full_oof[labels == 1], 0.01
        ),
        "global_far_5pct": global_far_operating_point(
            full_oof[labels == 0], full_oof[labels == 1], 0.05
        ),
    }
    results = {
        "selected_model": selected_name,
        "selection_metric": "5-fold GroupKFold out-of-fold AUC by claimed user",
        "model_comparison": comparison,
        "stages": stage_results,
        "features_by_stage": stages,
        "model_settings": {
            "group_kfold_splits": 5,
            "gradient_boosting_estimators": 100,
            "gradient_boosting_learning_rate": 0.05,
            "gradient_boosting_max_depth": 2,
        },
        "operating_points": operating_points,
        "decision_threshold": stage_results[
            "behavior_plus_deviations_plus_synthetic"
        ]["eer_threshold"],
        "cv": "5-fold GroupKFold by claimed user",
        "synthetic_device_config": config,
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": full_model,
            "threshold": results["decision_threshold"],
            "features": ALL_FEATURES,
            "selected_model": selected_name,
        },
        artifact_dir / "risk_model.joblib",
    )
    return results, frame, full_model, shap_values
