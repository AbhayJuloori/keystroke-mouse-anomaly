"""Export the numbers and small model parameters the project page needs.

Reads docs/results.json and the trained artifacts, writes site/data/site.json.
Run after `make train`.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _round(value, digits=4):
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {key: _round(item, digits) for key, item in value.items()}
    if isinstance(value, list):
        return [_round(item, digits) for item in value]
    return value


def _histogram(genuine: np.ndarray, impostor: np.ndarray, bins: int = 48) -> dict:
    upper = float(np.quantile(np.concatenate([genuine, impostor]), 0.99))
    edges = np.linspace(0.0, upper, bins + 1)
    return {
        "edges": edges.tolist(),
        "genuine": np.histogram(np.clip(genuine, 0, upper), edges)[0].tolist(),
        "impostor": np.histogram(np.clip(impostor, 0, upper), edges)[0].tolist(),
    }


def main() -> None:
    results = json.loads((ROOT / "docs/results.json").read_text(encoding="utf-8"))
    keystroke = joblib.load(ROOT / "artifacts/keystroke_models.joblib")
    pooled = np.load(ROOT / "artifacts/keystroke_pooled_scores.npz")
    sessions = pd.read_json(ROOT / "artifacts/scored_sessions.jsonl", lines=True)

    templates = {
        user: {
            "center": np.round(model["center"], 5).tolist(),
            "scale": np.round(model["scale"], 5).tolist(),
            "threshold": round(float(model["threshold"]), 5),
        }
        for user, model in keystroke["models"].items()
    }

    risk_column = "risk_behavior_plus_deviations_plus_synthetic"
    examples = []
    picks = [
        ("Genuine, low risk", sessions[sessions["label"] == 0].nsmallest(2, risk_column)),
        ("Impostor, high risk", sessions[sessions["label"] == 1].nlargest(2, risk_column)),
        ("Impostor that slipped through", sessions[sessions["label"] == 1].nsmallest(1, risk_column)),
        ("Genuine, wrongly flagged", sessions[sessions["label"] == 0].nlargest(1, risk_column)),
    ]
    for title, frame in picks:
        for _, row in frame.iterrows():
            examples.append(
                {
                    "title": title,
                    "user_id": row["user_id"],
                    "session_id": row["session_id"],
                    "label": int(row["label"]),
                    "risk": round(float(row[risk_column]), 1),
                    "behavioral_anomaly": round(float(row["behavioral_anomaly"]), 3),
                    "actions": int(row["action_count"]),
                    "flags": {
                        "emulator": int(row["synthetic_emulator"]),
                        "rooted_device": int(row["synthetic_rooted_device"]),
                        "vpn": int(row["synthetic_vpn"]),
                    },
                    "reasons": list(row["reason_codes"]),
                }
            )

    detectors = ["scaled_manhattan", "isolation_forest", "one_class_svm", "autoencoder", "fusion"]
    drift = results["drift_demo"]
    site = {
        "protocol": _round(results["protocol"]),
        "keystroke": {
            "feature_names": keystroke["feature_names"],
            "templates": templates,
            "summary": {name: _round(results["keystroke"][name]) for name in detectors},
            "per_user_eer": {
                name: [round(row["eer"], 4) for row in results["keystroke_per_user"][name]]
                for name in detectors
                if name in results["keystroke_per_user"]
            },
            "scaled_manhattan_histogram": _round(
                _histogram(
                    pooled["scaled_manhattan_genuine"], pooled["scaled_manhattan_impostor"]
                ),
                4,
            ),
            "paper_scaled_manhattan_eer": 0.096,
        },
        "mouse": _round(
            {
                key: results["mouse"][key]
                for key in (
                    "global_auc",
                    "global_eer",
                    "mean_user_auc",
                    "mean_user_eer",
                    "labeled_sessions",
                    "fixed_train_time_holdout_p95",
                    "operating_points",
                    "later_portion_only",
                    "eer_vs_actions",
                )
            }
        ),
        "mouse_per_user": _round(results["mouse_per_user"]),
        "risk": _round(
            {
                "stages": results["risk"]["stages"],
                "selected_model": results["risk"]["selected_model"],
                "model_comparison": results["risk"]["model_comparison"],
                "operating_points": results["risk"]["operating_points"],
                "synthetic_device_config": results["risk"]["synthetic_device_config"],
                "features": results["risk"]["features_by_stage"][
                    "behavior_plus_deviations_plus_synthetic"
                ],
            }
        ),
        "risk_examples": examples,
        "time_split": _round(results["time_split"]),
        "drift": _round(
            {
                "psi_threshold": drift["psi_threshold"],
                "aging": drift["keystroke_genuine_score_to_reference_median_ratio_by_session"],
                "null_psi": drift["null_psi_at_minimum_window"],
                "keystroke": {
                    key: value
                    for key, value in drift["keystroke"].items()
                    if key != "points"
                },
                "mouse": {
                    key: value for key, value in drift["mouse"].items() if key != "points"
                },
                "keystroke_points": [
                    [p["user_id"], p["requests_replayed"], round(p["psi"], 4)]
                    for p in drift["keystroke"]["points"]
                ],
                "mouse_points": [
                    [p["user_id"], p["sessions_replayed"], p["phase"][0], round(p["psi"], 4)]
                    for p in drift["mouse"]["points"]
                ],
            }
        ),
        "api_latency": _round(results["api_latency"], 2),
        "mlflow_metric_keys": results["mlflow_metric_verification"]["runs"],
    }
    out = ROOT / "site/data/site.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(site, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
