"""Per-user keystroke anomaly models and the CMU benchmark protocol."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from torch import nn

from .metrics import equal_error_rate, far_frr_at_threshold, global_far_operating_point


META_COLUMNS = ["subject", "sessionIndex", "rep"]


def protocol_split(
    data: pd.DataFrame, subject: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply the Killourhy-Maxion time split for one claimed subject."""
    features = [column for column in data.columns if column not in META_COLUMNS]
    own = data[data["subject"] == subject].sort_values(["sessionIndex", "rep"])
    train = own[own["sessionIndex"].between(1, 4)]
    genuine = own[own["sessionIndex"].between(5, 8)]
    impostor = (
        data[data["subject"] != subject]
        .sort_values(["subject", "sessionIndex", "rep"])
        .groupby("subject", sort=True)
        .head(5)
    )
    if len(train) != 200 or len(genuine) != 200 or len(impostor) != 250:
        raise ValueError(
            f"Unexpected protocol counts for {subject}: "
            f"{len(train)}, {len(genuine)}, {len(impostor)}"
        )
    return train[META_COLUMNS + features], genuine[META_COLUMNS + features], impostor[META_COLUMNS + features]


class ScaledManhattan:
    """Mean absolute distance scaled by the user's feature deviations."""

    def fit(self, values: np.ndarray) -> "ScaledManhattan":
        values = np.asarray(values, dtype=float)
        self.center_ = values.mean(axis=0)
        self.scale_ = values.std(axis=0, ddof=1)
        self.scale_[self.scale_ < 1e-9] = 1.0
        return self

    def score_samples(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        return np.mean(np.abs(values - self.center_) / self.scale_, axis=1)


class Autoencoder(nn.Module):
    def __init__(self, dimensions: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dimensions, 16),
            nn.ReLU(),
            nn.Linear(16, 6),
            nn.ReLU(),
            nn.Linear(6, 16),
            nn.ReLU(),
            nn.Linear(16, dimensions),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


def _fit_autoencoder(
    train: np.ndarray, seed: int, epochs: int = 40
) -> tuple[Autoencoder, StandardScaler]:
    torch.manual_seed(seed)
    scaler = StandardScaler().fit(train)
    tensor = torch.tensor(scaler.transform(train), dtype=torch.float32)
    model = Autoencoder(train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(tensor), tensor)
        loss.backward()
        optimizer.step()
    return model.eval(), scaler


def _autoencoder_scores(
    model: Autoencoder, scaler: StandardScaler, values: np.ndarray
) -> np.ndarray:
    tensor = torch.tensor(scaler.transform(values), dtype=torch.float32)
    with torch.no_grad():
        reconstructed = model(tensor)
    return ((reconstructed - tensor) ** 2).mean(dim=1).numpy()


def _summary(per_user: list[dict[str, Any]]) -> dict[str, float]:
    summary = {
        "mean_eer": float(np.mean([row["eer"] for row in per_user])),
        "std_eer": float(np.std([row["eer"] for row in per_user])),
    }
    for percentile in (90, 95, 99):
        summary[f"mean_far_at_train_p{percentile}"] = float(
            np.mean([row[f"far_at_train_p{percentile}"] for row in per_user])
        )
        summary[f"mean_frr_at_train_p{percentile}"] = float(
            np.mean([row[f"frr_at_train_p{percentile}"] for row in per_user])
        )
    return summary


def normalized_score_fusion(
    training_scores: dict[str, np.ndarray], scores: dict[str, np.ndarray]
) -> np.ndarray:
    """Normalize detector scores to enrollment distributions and average them."""
    if training_scores.keys() != scores.keys() or not scores:
        raise ValueError("Fusion requires matching non-empty detector score sets")
    normalized = []
    for name in training_scores:
        reference = np.asarray(training_scores[name], dtype=float)
        values = np.asarray(scores[name], dtype=float)
        scale = float(np.std(reference, ddof=1))
        if not np.isfinite(scale) or scale < 1e-9:
            scale = 1.0
        normalized.append((values - float(np.mean(reference))) / scale)
    return np.mean(np.vstack(normalized), axis=0)


def _out_of_sample_manhattan_reference(
    enrollment: pd.DataFrame, feature_names: list[str]
) -> np.ndarray:
    """Score each enrollment session with a model fit on the other sessions."""
    scores = []
    for session in sorted(enrollment["sessionIndex"].unique()):
        fitted = enrollment[enrollment["sessionIndex"] != session]
        held_out = enrollment[enrollment["sessionIndex"] == session]
        model = ScaledManhattan().fit(fitted[feature_names].to_numpy(dtype=float))
        scores.extend(model.score_samples(held_out[feature_names].to_numpy(dtype=float)))
    return np.asarray(scores, dtype=float)


def train_keystroke_models(
    csv_path: Path, artifact_dir: Path, seed: int = 42
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """Train and evaluate the four per-user keystroke detectors."""
    torch.set_num_threads(1)
    data = pd.read_csv(csv_path)
    feature_names = [column for column in data.columns if column not in META_COLUMNS]
    subjects = sorted(data["subject"].unique())
    per_detector: dict[str, list[dict[str, Any]]] = {
        "scaled_manhattan": [],
        "isolation_forest": [],
        "one_class_svm": [],
        "autoencoder": [],
        "fusion": [],
    }
    serving_models: dict[str, Any] = {}
    ae_states: dict[str, Any] = {}
    pooled: dict[str, dict[str, list[float]]] = {
        name: {"genuine": [], "impostor": []} for name in per_detector
    }

    for subject_index, subject in enumerate(subjects):
        train_df, genuine_df, impostor_df = protocol_split(data, subject)
        train = train_df[feature_names].to_numpy(dtype=float)
        genuine = genuine_df[feature_names].to_numpy(dtype=float)
        impostor = impostor_df[feature_names].to_numpy(dtype=float)

        manhattan = ScaledManhattan().fit(train)
        scaler_if = StandardScaler().fit(train)
        isolation = IsolationForest(
            n_estimators=120,
            contamination="auto",
            random_state=seed + subject_index,
            n_jobs=-1,
        ).fit(scaler_if.transform(train))
        scaler_svm = StandardScaler().fit(train)
        svm = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale").fit(
            scaler_svm.transform(train)
        )
        ae, scaler_ae = _fit_autoencoder(train, seed + subject_index)

        score_functions = {
            "scaled_manhattan": lambda values: manhattan.score_samples(values),
            "isolation_forest": lambda values: -isolation.score_samples(
                scaler_if.transform(values)
            ),
            "one_class_svm": lambda values: -svm.decision_function(
                scaler_svm.transform(values)
            ).ravel(),
            "autoencoder": lambda values: _autoencoder_scores(ae, scaler_ae, values),
        }
        subject_scores: dict[str, dict[str, np.ndarray]] = {}
        for name, score_function in score_functions.items():
            train_scores = score_function(train)
            genuine_scores = score_function(genuine)
            impostor_scores = score_function(impostor)
            subject_scores[name] = {
                "train": train_scores,
                "genuine": genuine_scores,
                "impostor": impostor_scores,
            }
        subject_scores["fusion"] = {
            split: normalized_score_fusion(
                {name: values["train"] for name, values in subject_scores.items()},
                {name: values[split] for name, values in subject_scores.items()},
            )
            for split in ("train", "genuine", "impostor")
        }

        for name, split_scores in subject_scores.items():
            train_scores = split_scores["train"]
            genuine_scores = split_scores["genuine"]
            impostor_scores = split_scores["impostor"]
            eer, eer_threshold, _, _ = equal_error_rate(genuine_scores, impostor_scores)
            row = {"user_id": subject, "eer": eer, "eer_threshold": eer_threshold}
            for percentile in (90, 95, 99):
                threshold = float(np.quantile(train_scores, percentile / 100.0))
                far, frr = far_frr_at_threshold(genuine_scores, impostor_scores, threshold)
                row[f"train_p{percentile}_threshold"] = threshold
                row[f"far_at_train_p{percentile}"] = far
                row[f"frr_at_train_p{percentile}"] = frr
            per_detector[name].append(row)
            pooled[name]["genuine"].extend(genuine_scores.tolist())
            pooled[name]["impostor"].extend(impostor_scores.tolist())

        train_scores = manhattan.score_samples(train)
        serving_models[subject] = {
            "center": manhattan.center_,
            "scale": manhattan.scale_,
            "threshold": float(np.quantile(train_scores, 0.95)),
            "reference_scores": _out_of_sample_manhattan_reference(
                train_df, feature_names
            ),
            "reference_source": "sessions_1_4_leave_one_session_out",
        }
        ae_states[subject] = {
            "state_dict": ae.state_dict(),
            "scaler_mean": scaler_ae.mean_,
            "scaler_scale": scaler_ae.scale_,
        }

    results: dict[str, Any] = {}
    for name, rows in per_detector.items():
        summary = _summary(rows)
        impostor_scores = np.asarray(pooled[name]["impostor"])
        genuine_scores = np.asarray(pooled[name]["genuine"])
        summary["operating_points"] = {
            "global_far_1pct": global_far_operating_point(
                genuine_scores, impostor_scores, 0.01
            ),
            "global_far_5pct": global_far_operating_point(
                genuine_scores, impostor_scores, 0.05
            ),
        }
        results[name] = summary

    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"feature_names": feature_names, "models": serving_models},
        artifact_dir / "keystroke_models.joblib",
    )
    torch.save(ae_states, artifact_dir / "keystroke_autoencoders.pt")
    np.savez_compressed(
        artifact_dir / "keystroke_pooled_scores.npz",
        **{
            f"{name}_{kind}": np.asarray(values[kind], dtype=float)
            for name, values in pooled.items()
            for kind in ("genuine", "impostor")
        },
    )
    (artifact_dir / "keystroke_per_user.json").write_text(
        json.dumps(per_detector, indent=2), encoding="utf-8"
    )
    return results, per_detector
