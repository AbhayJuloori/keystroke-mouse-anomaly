import joblib
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from kma.api import create_app
from kma.mouse import ACTION_FEATURES, DEVIATION_FEATURES, DEVIATION_SOURCES
from kma.risk import ALL_FEATURES, RiskModel


def _fixture_artifacts(path):
    joblib.dump(
        {
            "feature_names": ["f1", "f2"],
            "models": {
                "s001": {
                    "center": np.array([0.0, 0.0]),
                    "scale": np.array([1.0, 2.0]),
                    "threshold": 1.0,
                    "reference_scores": np.linspace(0.0, 1.0, 20),
                }
            },
        },
        path / "keystroke_models.joblib",
    )
    rng = np.random.default_rng(4)
    action_matrix = pd.DataFrame(rng.normal(size=(40, len(ACTION_FEATURES))), columns=ACTION_FEATURES)
    target = np.array([0] * 20 + [1] * 20)
    mouse_model = ExtraTreesClassifier(n_estimators=10, random_state=4).fit(action_matrix, target)
    joblib.dump(
        {
            "feature_names": ACTION_FEATURES,
            "models": {"user1": mouse_model},
            "reference_scores": {"user1": np.linspace(0.1, 0.5, 10)},
            "deviation_profiles": {
                "user1": {
                    name: {"source": source, "center": 0.0, "scale": 1.0}
                    for name, source in DEVIATION_SOURCES.items()
                }
            },
        },
        path / "mouse_models.joblib",
    )
    risk_frame = pd.DataFrame(
        {
            "behavioral_anomaly": [0.1, 0.2, 0.8, 0.9],
            **{name: [0.0, 0.1, 1.0, 1.1] for name in DEVIATION_FEATURES},
            "synthetic_emulator": [0, 0, 1, 1],
            "synthetic_rooted_device": [0, 1, 0, 1],
            "synthetic_vpn": [0, 0, 1, 1],
        }
    )
    scaler = StandardScaler().fit(risk_frame.to_numpy())
    logistic = LogisticRegression().fit(
        scaler.transform(risk_frame.to_numpy()), [0, 0, 1, 1]
    )
    risk_model = RiskModel(logistic, ALL_FEATURES, "logistic_regression", scaler)
    joblib.dump(
        {"model": risk_model, "threshold": 0.5, "features": ALL_FEATURES},
        path / "risk_model.joblib",
    )
    joblib.dump(
        {
            "reference_scores": {
                "s001": np.linspace(0.0, 1.0, 20),
                "user1": np.linspace(0.1, 0.5, 20),
            },
            "reference_sources": {
                "s001": "sessions_1_4_leave_one_session_out",
                "user1": "training_session_time_holdout_action_scores",
            },
            "reference_channels": {"s001": "keystroke", "user1": "mouse"},
            "drift_config": {
                "bins": 10,
                "smoothing": 0.5,
                "psi_threshold": 0.2,
                "channels": {
                    "keystroke": {"window_size": 200, "minimum_window": 50},
                    "mouse": {"window_size": 3, "minimum_window": 2},
                },
            },
        },
        path / "api_metadata.joblib",
    )


def test_api_endpoints_with_small_artifacts(tmp_path):
    _fixture_artifacts(tmp_path)
    client = TestClient(create_app(tmp_path))
    assert client.get("/health").json()["status"] == "ok"
    key_response = client.post(
        "/score/keystroke",
        json={"user_id": "s001", "features": {"f1": 0.5, "f2": 1.0}},
    )
    assert key_response.status_code == 200
    assert key_response.json()["decision"] in {"accept", "anomaly"}
    events = [
        {"record_timestamp": 0.0, "x": 0, "y": 0},
        {"record_timestamp": 0.1, "x": 2, "y": 1},
        {"record_timestamp": 0.2, "x": 4, "y": 2},
        {"record_timestamp": 0.3, "x": 6, "y": 3},
    ]
    response = client.post(
        "/score/session",
        json={"user_id": "user1", "events": events, "synthetic_vpn": True},
    )
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["risk_score"] <= 100
    assert len(body["top_reason_codes"]) == 3
    assert set(body["deviation_features"]) == set(DEVIATION_FEATURES)
    assert body["action_count"] == 1
    mouse_drift = client.get("/drift/user1")
    assert mouse_drift.status_code == 200
    # Mouse drift buffers one anomaly score per action, not one per session.
    assert mouse_drift.json()["channel"] == "mouse"
    assert mouse_drift.json()["recent_score_count"] == 1
    assert not mouse_drift.json()["enough_data"]
    assert (
        mouse_drift.json()["reference_source"]
        == "training_session_time_holdout_action_scores"
    )
    for _ in range(3):
        client.post("/score/session", json={"user_id": "user1", "events": events})
    mouse_drift = client.get("/drift/user1").json()
    assert mouse_drift["recent_score_count"] == 3
    assert mouse_drift["window_size"] == 3
    assert mouse_drift["enough_data"]
    key_drift = client.get("/drift/s001")
    assert key_drift.status_code == 200
    assert key_drift.json()["reference_source"] == "sessions_1_4_leave_one_session_out"
