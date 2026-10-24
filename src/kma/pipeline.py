"""End-to-end training, evaluation, tracking, artifact, and report pipeline."""

from __future__ import annotations

import os
import time
import json
from pathlib import Path
from typing import Any

os.environ.setdefault("MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING", "false")
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from fastapi.testclient import TestClient

from .keystroke import META_COLUMNS, protocol_split, train_keystroke_models
from .mouse import train_mouse_models
from .report import create_figures, save_results, write_documentation
from .risk import DEVICE_FEATURES, train_risk_model


ROOT = Path(__file__).resolve().parents[2]


def _log_keystroke_runs(
    results: dict[str, Any],
    per_user: dict[str, list[dict[str, Any]]],
    pipeline_run_id: str,
) -> None:
    mlflow.set_experiment("keystroke-anomaly")
    for detector, metrics in results.items():
        with mlflow.start_run(run_name=detector):
            mlflow.set_tag("pipeline_run_id", pipeline_run_id)
            mlflow.set_tag("run_type", "keystroke")
            mlflow.log_params(
                {
                    "detector": detector,
                    "split": "sessions_1_4_train_5_8_test",
                    "seed": 42,
                }
            )
            mlflow.log_metrics(_numeric_metrics(metrics))
            mlflow.log_dict(per_user[detector], "per_user_metrics.json")


def _numeric_metrics(values: dict[str, Any], prefix: str = "") -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in values.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            metrics.update(_numeric_metrics(value, name))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            metrics[name] = float(value)
    return metrics


def _log_mouse_risk_runs(
    mouse_results: dict[str, Any],
    mouse_per_user: list[dict[str, Any]],
    risk_results: dict[str, Any],
    pipeline_run_id: str,
) -> None:
    mlflow.set_experiment("mouse-risk")
    with mlflow.start_run(run_name="mouse-action-extra-trees"):
        mlflow.set_tag("pipeline_run_id", pipeline_run_id)
        mlflow.set_tag("run_type", "mouse")
        mlflow.log_params(
            {
                "model": "per_user_extra_trees",
                "n_estimators": 120,
                "max_actions_per_session": 250,
                "test_scope": "public_labels_only",
            }
        )
        mlflow.log_metrics(_numeric_metrics(mouse_results))
        mlflow.log_dict(mouse_per_user, "per_user_metrics.json")
    with mlflow.start_run(run_name=f"session-risk-{risk_results['selected_model']}"):
        mlflow.set_tag("pipeline_run_id", pipeline_run_id)
        mlflow.set_tag("run_type", "risk")
        mlflow.log_params(
            {
                "model": risk_results["selected_model"],
                "cv": risk_results["cv"],
                "device_signals": "SYNTHETIC",
            }
        )
        mlflow.log_metrics(_numeric_metrics(risk_results))
        mlflow.log_dict(risk_results["synthetic_device_config"], "synthetic_device_config.json")


def null_psi(bins: int, window: int, reference: int) -> float:
    """Approximate expected PSI from sampling noise alone (no real change)."""
    return (bins - 1) * (1.0 / window + 1.0 / reference)


def minimum_drift_window(bins: int, reference: int, threshold: float) -> int:
    """Smallest window whose no-change PSI is at most half the alert threshold."""
    limit = threshold / 2.0 / (bins - 1) - 1.0 / reference
    if limit <= 0:
        raise ValueError("Reference is too small for this PSI threshold")
    return int(np.ceil(1.0 / limit))


DRIFT_CONFIG = {
    "bins": 10,
    "smoothing": 0.5,
    "psi_threshold": 0.2,
    "channels": {
        # 200 leave-one-session-out reference scores per CMU user gives a
        # minimum of 164; the window is the user's whole genuine test block.
        "keystroke": {"window_size": 200, "minimum_window": minimum_drift_window(10, 200, 0.2)},
        "mouse": {"window_size": 2000, "minimum_window": 200},
    },
}


def _write_api_metadata(artifact_dir: Path) -> None:
    """Save per-user drift references: unseen genuine scores in the served unit.

    Keystroke drift watches per-sample scaled Manhattan scores against
    leave-one-session-out scores from sessions 1-4. Mouse drift watches
    per-action anomaly scores against the later-in-time hold-out of each
    training session, which gives hundreds of reference values per user
    instead of the five to seven session scores.
    """
    keystroke = joblib.load(artifact_dir / "keystroke_models.joblib")
    mouse = joblib.load(artifact_dir / "mouse_models.joblib")
    references: dict[str, np.ndarray] = {}
    sources: dict[str, str] = {}
    channels: dict[str, str] = {}
    for user, model in keystroke["models"].items():
        references[user] = np.asarray(model["reference_scores"], dtype=float)
        sources[user] = model["reference_source"]
        channels[user] = "keystroke"
    for user, scores in mouse["reference_action_scores"].items():
        references[user] = np.asarray(scores, dtype=float)
        sources[user] = "training_session_time_holdout_action_scores"
        channels[user] = "mouse"
    joblib.dump(
        {
            "reference_scores": references,
            "reference_sources": sources,
            "reference_channels": channels,
            "drift_config": DRIFT_CONFIG,
        },
        artifact_dir / "api_metadata.joblib",
    )


def _verify_mlflow_metrics(pipeline_run_id: str) -> dict[str, Any]:
    """Read the just-written file-backed MLflow runs and assert metric keys."""
    required = {
        "keystroke": [
            *[
                f"mean_{rate}_at_train_p{percentile}"
                for percentile in (90, 95, 99)
                for rate in ("far", "frr")
            ],
            *[
                f"operating_points.global_far_{target}pct.{rate}"
                for target in (1, 5)
                for rate in ("far", "frr")
            ],
        ],
        "mouse": [
            "fixed_train_time_holdout_p95.far",
            "fixed_train_time_holdout_p95.frr",
            "operating_points.global_far_1pct.far",
            "operating_points.global_far_1pct.frr",
            "operating_points.global_far_5pct.far",
            "operating_points.global_far_5pct.frr",
        ],
        "risk": [
            "operating_points.global_far_1pct.far",
            "operating_points.global_far_1pct.frr",
            "operating_points.global_far_5pct.far",
            "operating_points.global_far_5pct.frr",
        ],
    }
    verified: dict[str, Any] = {"pipeline_run_id": pipeline_run_id, "runs": {}}
    for experiment_name, run_types in {
        "keystroke-anomaly": ("keystroke",),
        "mouse-risk": ("mouse", "risk"),
    }.items():
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise AssertionError(f"Missing MLflow experiment: {experiment_name}")
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.pipeline_run_id = '{pipeline_run_id}'",
        )
        for run_type in run_types:
            selected = runs[runs["tags.run_type"] == run_type]
            if selected.empty:
                raise AssertionError(f"Missing MLflow {run_type} run")
            expected = required[run_type]
            for _, run in selected.iterrows():
                missing = [key for key in expected if f"metrics.{key}" not in run.index]
                if missing:
                    raise AssertionError(
                        f"MLflow run {run['tags.mlflow.runName']} lacks metrics: {missing}"
                    )
            verified["runs"][run_type] = {
                "run_count": int(len(selected)),
                "metric_keys": expected,
            }
    return verified


def _request_latency(client: TestClient, path: str, payload: dict[str, Any]) -> dict[str, float]:
    for _ in range(5):
        response = client.post(path, json=payload)
        response.raise_for_status()
    timings = []
    for _ in range(200):
        started = time.perf_counter()
        response = client.post(path, json=payload)
        response.raise_for_status()
        timings.append((time.perf_counter() - started) * 1000.0)
    return {
        "request_count": 200,
        "p50_ms": float(np.percentile(timings, 50)),
        "p95_ms": float(np.percentile(timings, 95)),
    }


def _api_measurements(artifact_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Measure API latency and replay genuine then impostor traffic per user."""
    from .api import create_app

    keystrokes = pd.read_csv(ROOT / "data/raw/DSL-StrongPasswordData.csv")
    feature_names = [column for column in keystrokes if column not in META_COLUMNS]
    subjects = sorted(keystrokes["subject"].unique())[:5]
    subject = subjects[0]
    _, genuine, _ = protocol_split(keystrokes, subject)
    key_payload = {
        "user_id": subject,
        "features": genuine.iloc[0][feature_names].astype(float).to_dict(),
    }

    labels = pd.read_csv(ROOT / "data/raw/balabit/public_labels.csv")
    labeled_names = set(labels["filename"])
    mouse_path = next(
        path
        for path in sorted((ROOT / "data/raw/balabit/test_files").glob("user*/*"))
        if path.name in labeled_names
    )
    mouse_events = pd.read_csv(mouse_path).head(250)
    event_payload = json.loads(
        mouse_events.rename(
            columns={
                "record timestamp": "record_timestamp",
                "client timestamp": "client_timestamp",
            }
        ).to_json(orient="records")
    )
    session_payload = {"user_id": mouse_path.parent.name, "events": event_payload}
    latency_client = TestClient(create_app(artifact_dir))
    latency = {
        "keystroke": _request_latency(
            latency_client, "/score/keystroke", key_payload
        ),
        "session": _request_latency(latency_client, "/score/session", session_payload),
    }

    keystroke_points = []
    block_size = 50
    for subject in sorted(keystrokes["subject"].unique()):
        drift_client = TestClient(create_app(artifact_dir))
        _, subject_genuine, subject_impostor = protocol_split(keystrokes, subject)
        request_count = 0
        for phase, frame in (
            ("genuine", subject_genuine),
            ("impostor", subject_impostor),
        ):
            records = frame.reset_index(drop=True)
            for start in range(0, len(records), block_size):
                block = records.iloc[start : start + block_size]
                for _, row in block.iterrows():
                    response = drift_client.post(
                        "/score/keystroke",
                        json={
                            "user_id": subject,
                            "features": row[feature_names].astype(float).to_dict(),
                        },
                    )
                    response.raise_for_status()
                    request_count += 1
                body = drift_client.get(f"/drift/{subject}").json()
                if not body["enough_data"]:
                    continue
                keystroke_points.append(
                    {
                        "user_id": subject,
                        "phase": phase,
                        "requests_replayed": request_count,
                        "recent_score_count": int(body["recent_score_count"]),
                        "psi": float(body["psi"]),
                        "drift": bool(body["drift"]),
                        "window_pure_impostor": bool(
                            phase == "impostor"
                            and request_count - len(subject_genuine)
                            >= DRIFT_CONFIG["channels"]["keystroke"]["window_size"]
                        ),
                    }
                )

    mouse_points = []
    labels_by_name = labels.set_index("filename")["is_illegal"]
    mouse_paths = [
        path
        for path in sorted((ROOT / "data/raw/balabit/test_files").glob("user*/*"))
        if path.name in labeled_names
    ]
    for user in sorted({path.parent.name for path in mouse_paths}):
        drift_client = TestClient(create_app(artifact_dir))
        own = [path for path in mouse_paths if path.parent.name == user]
        sessions_replayed = 0
        impostor_actions = 0
        for phase, label in (("genuine", 0), ("impostor", 1)):
            for path in [p for p in own if int(labels_by_name[p.name]) == label]:
                events = json.loads(
                    pd.read_csv(path)
                    .rename(
                        columns={
                            "record timestamp": "record_timestamp",
                            "client timestamp": "client_timestamp",
                        }
                    )
                    .to_json(orient="records")
                )
                response = drift_client.post(
                    "/score/session", json={"user_id": user, "events": events}
                )
                response.raise_for_status()
                sessions_replayed += 1
                if phase == "impostor":
                    impostor_actions += int(response.json()["action_count"])
                body = drift_client.get(f"/drift/{user}").json()
                if not body["enough_data"]:
                    continue
                mouse_points.append(
                    {
                        "user_id": user,
                        "phase": phase,
                        "sessions_replayed": sessions_replayed,
                        "recent_score_count": int(body["recent_score_count"]),
                        "psi": float(body["psi"]),
                        "drift": bool(body["drift"]),
                        "window_pure_impostor": bool(
                            impostor_actions
                            >= DRIFT_CONFIG["channels"]["mouse"]["window_size"]
                        ),
                    }
                )

    def summarize(points: list[dict[str, Any]]) -> dict[str, Any]:
        users = sorted({point["user_id"] for point in points})
        summaries = {}
        for threshold in (0.1, 0.2, 0.25):
            genuine = [p["psi"] > threshold for p in points if p["phase"] == "genuine"]
            impostor = [p["psi"] > threshold for p in points if p["phase"] == "impostor"]
            pure = [
                p["psi"] > threshold for p in points if p["window_pure_impostor"]
            ]
            genuine_users = [
                any(p["psi"] > threshold for p in points if p["user_id"] == user and p["phase"] == "genuine")
                for user in users
            ]
            impostor_users = [
                any(p["psi"] > threshold for p in points if p["user_id"] == user and p["phase"] == "impostor")
                for user in users
            ]
            summaries[str(threshold)] = {
                "genuine_checkpoint_count": len(genuine),
                "false_drift_rate": float(np.mean(genuine)) if genuine else None,
                "impostor_checkpoint_count": len(impostor),
                "detection_rate": float(np.mean(impostor)) if impostor else None,
                "pure_impostor_checkpoint_count": len(pure),
                "pure_impostor_detection_rate": float(np.mean(pure)) if pure else None,
                "users_with_any_false_drift": int(sum(genuine_users)),
                "users_with_impostor_drift_flag": int(sum(impostor_users)),
            }
        return {
            "user_count": len(users),
            "median_genuine_psi": float(np.median([p["psi"] for p in points if p["phase"] == "genuine"])),
            "median_impostor_psi": float(np.median([p["psi"] for p in points if p["phase"] == "impostor"])),
            "threshold_summaries": summaries,
        }

    keystroke_bundle = joblib.load(artifact_dir / "keystroke_models.joblib")
    aging_rows = []
    for subject, model in keystroke_bundle["models"].items():
        own = keystrokes[keystrokes["subject"] == subject]
        reference_median = float(np.median(model["reference_scores"]))
        for session in range(1, 9):
            values = own.loc[own["sessionIndex"] == session, feature_names].to_numpy(dtype=float)
            scores = np.mean(np.abs(values - model["center"]) / model["scale"], axis=1)
            aging_rows.append(
                {"session": session, "ratio": float(np.median(scores)) / reference_median}
            )
    aging = (
        pd.DataFrame(aging_rows).groupby("session")["ratio"].median().to_dict()
    )
    mouse_reference_sizes = [
        len(scores)
        for scores in joblib.load(artifact_dir / "mouse_models.joblib")[
            "reference_action_scores"
        ].values()
    ]
    drift_demo = {
        "null_psi_at_minimum_window": {
            "keystroke": null_psi(
                DRIFT_CONFIG["bins"],
                DRIFT_CONFIG["channels"]["keystroke"]["minimum_window"],
                200,
            ),
            "mouse_median_reference": null_psi(
                DRIFT_CONFIG["bins"],
                DRIFT_CONFIG["channels"]["mouse"]["minimum_window"],
                int(np.median(mouse_reference_sizes)),
            ),
        },
        "mouse_reference_actions_per_user": {
            "min": int(min(mouse_reference_sizes)),
            "median": float(np.median(mouse_reference_sizes)),
            "max": int(max(mouse_reference_sizes)),
        },
        "keystroke_genuine_score_to_reference_median_ratio_by_session": {
            str(int(key)): float(value) for key, value in aging.items()
        },
        "psi_bins": DRIFT_CONFIG["bins"],
        "psi_smoothing": DRIFT_CONFIG["smoothing"],
        "psi_threshold": DRIFT_CONFIG["psi_threshold"],
        "comparison_thresholds": [0.1, 0.2, 0.25],
        "keystroke": {
            "reference_source": "sessions_1_4_leave_one_session_out",
            "genuine_requests_per_user": 200,
            "impostor_requests_per_user": 250,
            "checkpoint_every_requests": block_size,
            **DRIFT_CONFIG["channels"]["keystroke"],
            **summarize(keystroke_points),
            "points": keystroke_points,
        },
        "mouse": {
            "reference_source": "training_session_time_holdout_action_scores",
            "unit": "per-action anomaly score, one checkpoint per replayed session",
            **DRIFT_CONFIG["channels"]["mouse"],
            **summarize(mouse_points),
            "points": mouse_points,
        },
    }
    return latency, drift_demo


def main() -> None:
    started = time.perf_counter()
    artifact_dir = ROOT / "artifacts"
    docs_dir = ROOT / "docs"
    artifact_dir.mkdir(exist_ok=True)
    docs_dir.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"file:{ROOT / 'mlruns'}")
    pipeline_run_id = str(time.time_ns())
    split_config = yaml.safe_load(
        (ROOT / "configs/split.yaml").read_text(encoding="utf-8")
    )

    keystroke_results, keystroke_per_user = train_keystroke_models(
        ROOT / "data/raw/DSL-StrongPasswordData.csv", artifact_dir
    )
    _log_keystroke_runs(
        keystroke_results, keystroke_per_user, pipeline_run_id
    )

    mouse_results, mouse_per_user, mouse_sessions, session_features = train_mouse_models(
        ROOT / "data/raw/balabit",
        artifact_dir,
        time_fraction=float(split_config["time_fraction"]),
        test_later_start_fraction=float(
            split_config["test_later_start_fraction"]
        ),
    )
    session_features.to_csv(artifact_dir / "mouse_session_features.csv", index=False)
    risk_results, risk_sessions, risk_model, shap_values = train_risk_model(
        mouse_sessions, ROOT / "configs/device_signals.yaml", artifact_dir
    )
    risk_sessions.drop(
        columns=["_action_anomaly_scores", "_later_action_anomaly_scores"]
    ).to_json(
        artifact_dir / "scored_sessions.jsonl",
        orient="records",
        lines=True,
    )
    _write_api_metadata(artifact_dir)
    _log_mouse_risk_runs(
        mouse_results,
        mouse_per_user,
        risk_results,
        pipeline_run_id,
    )
    mlflow_verification = _verify_mlflow_metrics(pipeline_run_id)
    api_latency, drift_demo = _api_measurements(artifact_dir)

    keystroke_data = pd.read_csv(ROOT / "data/raw/DSL-StrongPasswordData.csv")
    cmu_splits = [
        protocol_split(keystroke_data, subject)
        for subject in sorted(keystroke_data["subject"].unique())
    ]
    cmu_train_sessions = sorted(
        {
            int(session)
            for train, _, _ in cmu_splits
            for session in train["sessionIndex"].unique()
        }
    )
    cmu_test_sessions = sorted(
        {
            int(session)
            for _, genuine, _ in cmu_splits
            for session in genuine["sessionIndex"].unique()
        }
    )
    time_split = {
        "cmu": {
            "method": "by_sessionIndex",
            "enrollment_sessions": cmu_train_sessions,
            "genuine_test_sessions": cmu_test_sessions,
            "enrollment_samples_per_user": 200,
            "genuine_test_samples_per_user": 200,
            "leakage_checks": {
                "sessions_5_8_absent_from_train": bool(
                    not set(cmu_train_sessions).intersection({5, 6, 7, 8})
                ),
                "training_sessions_precede_genuine_test": bool(
                    max(cmu_train_sessions) < min(cmu_test_sessions)
                ),
            },
        },
        "balabit": mouse_results["time_split"],
    }

    results = {
        "title": "Keystroke and Mouse Anomaly Detection (CMU, Balabit data)",
        "seed": 42,
        "pipeline_runtime_seconds": 0.0,
        "protocol": {
            "keystroke_users": 51,
            "keystroke_total_samples": 20400,
            "keystroke_train_per_user": 200,
            "keystroke_genuine_test_per_user": 200,
            "keystroke_impostor_test_per_user": 250,
            "keystroke_feature_count": 31,
            "keystroke_isolation_forest_estimators": 120,
            "keystroke_autoencoder_epochs": 40,
            "keystroke_feature_names": joblib.load(
                artifact_dir / "keystroke_models.joblib"
            )["feature_names"],
            "mouse_labeled_sessions": int(len(mouse_sessions)),
            "mouse_users": int(mouse_sessions["user_id"].nunique()),
            "mouse_training_sessions": int(
                (session_features["partition"] == "training").sum()
            ),
            "mouse_deviation_feature_count": 8,
            "mouse_extra_trees_estimators": 120,
            "mouse_max_actions_per_training_session": 250,
            "action_pause_seconds": 10.0,
            "minimum_action_events": 4,
            "drift_psi_threshold": 0.2,
            "drift_windows": DRIFT_CONFIG["channels"],
            "drift_psi_bins": 10,
            "drift_psi_smoothing": 0.5,
            "api_latency_request_count_per_endpoint": 200,
            "balabit_split": "within-session elapsed-time split inside training_files",
            "balabit_cross_session_chronology_verified": False,
        },
        "time_split": time_split,
        "keystroke": keystroke_results,
        "keystroke_per_user": keystroke_per_user,
        "mouse": mouse_results,
        "mouse_per_user": mouse_per_user,
        "risk": risk_results,
        "api_latency": api_latency,
        "drift_demo": drift_demo,
        "mlflow_metric_verification": mlflow_verification,
    }
    create_figures(
        results,
        keystroke_per_user,
        mouse_per_user,
        risk_sessions,
        shap_values,
        artifact_dir,
        docs_dir / "figures",
    )
    elapsed = time.perf_counter() - started
    results["pipeline_runtime_seconds"] = float(elapsed)
    save_results(results, docs_dir / "results.json")
    write_documentation(results, ROOT)
    print(f"Pipeline completed in {elapsed:.2f} seconds")
    print(f"Results: {docs_dir / 'results.json'}")


if __name__ == "__main__":
    main()
