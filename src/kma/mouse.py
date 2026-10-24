"""Mouse action segmentation, feature engineering, and per-user models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score

from .metrics import equal_error_rate, far_frr_at_threshold, global_far_operating_point
from .splits import (
    ACTION_END,
    ACTION_START,
    ACTION_TIME_COLUMNS,
    SESSION_END,
    SESSION_START,
    time_split_actions,
)


ACTION_FEATURES = [
    "duration",
    "n_points",
    "path_length",
    "displacement",
    "straightness",
    "vx_mean",
    "vx_std",
    "vx_max_abs",
    "vy_mean",
    "vy_std",
    "vy_max_abs",
    "speed_mean",
    "speed_std",
    "speed_max",
    "acceleration_mean",
    "acceleration_std",
    "acceleration_max",
    "jerk_mean",
    "jerk_std",
    "jerk_max",
    "angular_velocity_mean",
    "angular_velocity_std",
    "angular_velocity_max",
    "curvature_mean",
    "curvature_std",
    "curvature_max",
    "click_duration",
    "direction_sin",
    "direction_cos",
    "is_move",
    "is_click",
    "is_drag",
]

DEVIATION_SOURCES = {
    "deviation_speed_mean": "speed_mean_mean",
    "deviation_speed_max": "speed_max_mean",
    "deviation_curvature_mean": "curvature_mean_mean",
    "deviation_straightness": "straightness_mean",
    "deviation_click_duration": "click_duration_mean",
    "deviation_acceleration_mean": "acceleration_mean_mean",
    "deviation_angular_velocity_mean": "angular_velocity_mean_mean",
    "deviation_action_duration": "duration_mean",
}
DEVIATION_FEATURES = list(DEVIATION_SOURCES)


def segment_actions(
    events: pd.DataFrame, pause_seconds: float = 10.0, min_events: int = 4
) -> list[tuple[str, pd.DataFrame]]:
    """Split a session at long pauses and completed click interactions."""
    if events.empty:
        return []
    frame = events.sort_values("record timestamp", kind="stable").reset_index(drop=True)
    segments: list[pd.DataFrame] = []
    start = 0
    for index in range(1, len(frame)):
        gap = float(frame.loc[index, "record timestamp"] - frame.loc[index - 1, "record timestamp"])
        previous_state = str(frame.loc[index - 1, "state"]).lower()
        if gap > pause_seconds or previous_state == "released":
            segments.append(frame.iloc[start:index].copy())
            start = index
    segments.append(frame.iloc[start:].copy())

    actions: list[tuple[str, pd.DataFrame]] = []
    for segment in segments:
        if len(segment) < min_events:
            continue
        states = segment["state"].astype(str).str.lower()
        if states.eq("drag").any():
            action_type = "drag_drop"
        elif states.isin(["pressed", "released"]).any():
            action_type = "point_click"
        else:
            action_type = "mouse_move"
        actions.append((action_type, segment.reset_index(drop=True)))
    return actions


def _stats(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 0.0, 0.0
    return float(values.mean()), float(values.std()), float(np.max(np.abs(values)))


def action_features(action_type: str, events: pd.DataFrame) -> dict[str, float]:
    """Compute kinematic and interaction features for one action."""
    t = events["record timestamp"].to_numpy(dtype=float)
    x = events["x"].to_numpy(dtype=float)
    y = events["y"].to_numpy(dtype=float)
    dt = np.diff(t)
    dx = np.diff(x)
    dy = np.diff(y)
    valid = dt > 1e-6
    dt, dx, dy = dt[valid], dx[valid], dy[valid]
    step = np.hypot(dx, dy)
    vx = np.divide(dx, dt, out=np.zeros_like(dx), where=dt > 0)
    vy = np.divide(dy, dt, out=np.zeros_like(dy), where=dt > 0)
    speed = np.hypot(vx, vy)

    if speed.size >= 2:
        second_dt = dt[1:]
        acceleration = np.divide(
            np.diff(speed), second_dt, out=np.zeros(speed.size - 1), where=second_dt > 0
        )
        angles = np.unwrap(np.arctan2(vy, vx))
        angle_delta = np.diff(angles)
        angular_velocity = np.divide(
            angle_delta,
            second_dt,
            out=np.zeros(angle_delta.size),
            where=second_dt > 0,
        )
        curvature = np.divide(
            np.abs(angle_delta),
            step[1:],
            out=np.zeros(angle_delta.size),
            where=step[1:] > 0,
        )
    else:
        acceleration = angular_velocity = curvature = np.array([], dtype=float)
    if acceleration.size >= 2:
        jerk_dt = dt[2:]
        jerk = np.divide(
            np.diff(acceleration),
            jerk_dt,
            out=np.zeros(acceleration.size - 1),
            where=jerk_dt > 0,
        )
    else:
        jerk = np.array([], dtype=float)

    displacement = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
    path_length = float(step.sum())
    direction = float(np.arctan2(y[-1] - y[0], x[-1] - x[0]))
    states = events["state"].astype(str).str.lower().to_numpy()
    pressed = np.flatnonzero(states == "pressed")
    released = np.flatnonzero(states == "released")
    click_duration = 0.0
    if pressed.size and released.size:
        releases_after = released[released >= pressed[0]]
        if releases_after.size:
            click_duration = max(0.0, float(t[releases_after[0]] - t[pressed[0]]))

    vx_mean, vx_std, vx_max = _stats(vx)
    vy_mean, vy_std, vy_max = _stats(vy)
    speed_mean, speed_std, speed_max = _stats(speed)
    acc_mean, acc_std, acc_max = _stats(acceleration)
    jerk_mean, jerk_std, jerk_max = _stats(jerk)
    angular_mean, angular_std, angular_max = _stats(angular_velocity)
    curve_mean, curve_std, curve_max = _stats(curvature)
    return {
        "duration": max(0.0, float(t[-1] - t[0])),
        "n_points": float(len(events)),
        "path_length": path_length,
        "displacement": displacement,
        "straightness": displacement / path_length if path_length > 0 else 0.0,
        "vx_mean": vx_mean,
        "vx_std": vx_std,
        "vx_max_abs": vx_max,
        "vy_mean": vy_mean,
        "vy_std": vy_std,
        "vy_max_abs": vy_max,
        "speed_mean": speed_mean,
        "speed_std": speed_std,
        "speed_max": speed_max,
        "acceleration_mean": acc_mean,
        "acceleration_std": acc_std,
        "acceleration_max": acc_max,
        "jerk_mean": jerk_mean,
        "jerk_std": jerk_std,
        "jerk_max": jerk_max,
        "angular_velocity_mean": angular_mean,
        "angular_velocity_std": angular_std,
        "angular_velocity_max": angular_max,
        "curvature_mean": curve_mean,
        "curvature_std": curve_std,
        "curvature_max": curve_max,
        "click_duration": click_duration,
        "direction_sin": float(np.sin(direction)),
        "direction_cos": float(np.cos(direction)),
        "is_move": float(action_type == "mouse_move"),
        "is_click": float(action_type == "point_click"),
        "is_drag": float(action_type == "drag_drop"),
    }


def extract_action_features(events: pd.DataFrame) -> pd.DataFrame:
    columns = ACTION_FEATURES + ACTION_TIME_COLUMNS
    if events.empty:
        return pd.DataFrame(columns=columns)
    session_start = float(events["record timestamp"].min())
    session_end = float(events["record timestamp"].max())
    rows = []
    for kind, frame in segment_actions(events):
        rows.append(
            {
                **action_features(kind, frame),
                ACTION_START: float(frame["record timestamp"].iloc[0]),
                ACTION_END: float(frame["record timestamp"].iloc[-1]),
                SESSION_START: session_start,
                SESSION_END: session_end,
            }
        )
    return pd.DataFrame(rows, columns=columns).replace([np.inf, -np.inf], 0.0).fillna(0.0)


def aggregate_session(actions: pd.DataFrame) -> dict[str, float]:
    """Aggregate action features into a session feature vector."""
    result: dict[str, float] = {"action_count": float(len(actions))}
    if actions.empty:
        for feature in ACTION_FEATURES:
            for suffix in ("mean", "q25", "q50", "q75"):
                result[f"{feature}_{suffix}"] = 0.0
        return result
    for feature in ACTION_FEATURES:
        values = actions[feature].to_numpy(dtype=float)
        result[f"{feature}_mean"] = float(np.mean(values))
        result[f"{feature}_q25"] = float(np.quantile(values, 0.25))
        result[f"{feature}_q50"] = float(np.quantile(values, 0.50))
        result[f"{feature}_q75"] = float(np.quantile(values, 0.75))
    return result


def fit_deviation_profile(training_sessions: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Fit per-feature median and MAD scales from one user's training sessions."""
    profile: dict[str, dict[str, float]] = {}
    for output_name, source_name in DEVIATION_SOURCES.items():
        values = training_sessions[source_name].to_numpy(dtype=float)
        center = float(np.median(values))
        mad_scale = 1.4826 * float(np.median(np.abs(values - center)))
        scale_floor = max(1e-6, 0.05 * max(abs(center), 1.0))
        profile[output_name] = {
            "source": source_name,
            "center": center,
            "scale": max(mad_scale, scale_floor),
        }
    return profile


def session_deviation_features(
    session_aggregate: dict[str, float] | pd.Series,
    profile: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Convert session aggregates to signed per-user deviation features."""
    return {
        name: (
            float(session_aggregate[settings["source"]]) - float(settings["center"])
        )
        / float(settings["scale"])
        for name, settings in profile.items()
    }


def _read_actions(path: Path) -> pd.DataFrame:
    return extract_action_features(pd.read_csv(path))


def _sample_session_actions(
    actions: pd.DataFrame, limit: int, seed: int
) -> pd.DataFrame:
    if len(actions) <= limit:
        return actions
    return actions.sample(limit, random_state=seed)


def load_mouse_training_records(
    training_paths: list[Path],
    seed: int = 42,
    max_actions_per_session: int = 250,
    time_fraction: float = 0.7,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read training files and expose only early actions to fitting paths."""
    records: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    for index, path in enumerate(training_paths):
        if "training_files" not in path.parts:
            raise ValueError(f"Mouse fitting path is not a training file: {path}")
        all_actions = _read_actions(path)
        enrollment, holdout, split = time_split_actions(all_actions, time_fraction)
        records.append(
            {
                "path": path,
                "user_id": path.parent.name,
                "session_id": path.name,
                "source_partition": "training_files",
                "actions": enrollment,
                "holdout_actions": holdout,
                "time_split": split,
                "sampled_actions": _sample_session_actions(
                    enrollment, max_actions_per_session, seed + index
                ),
            }
        )
        aggregate_rows.append(
            {
                "partition": "training",
                "user_id": path.parent.name,
                "session_id": path.name,
                **aggregate_session(enrollment),
            }
        )
    return records, aggregate_rows


def _fit_user_action_model(
    train_records: list[dict[str, Any]],
    user: str,
    seed: int,
    excluded_session: str | None = None,
) -> ExtraTreesClassifier:
    pieces, targets = [], []
    for record in train_records:
        if record.get("source_partition") != "training_files":
            raise ValueError("Mouse fitting received a non-training record")
        if record["session_id"] == excluded_session and record["user_id"] == user:
            continue
        actions = record["sampled_actions"]
        if actions.empty:
            continue
        split = record["time_split"]
        if not split["enrollment_starts_before_cut"] or not split[
            "enrollment_ends_at_or_before_cut"
        ]:
            raise ValueError("Mouse fitting received non-enrollment actions")
        pieces.append(actions[ACTION_FEATURES])
        targets.extend([int(record["user_id"] == user)] * len(actions))
    matrix = pd.concat(pieces, ignore_index=True)
    target = np.asarray(targets, dtype=int)
    return ExtraTreesClassifier(
        n_estimators=120,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    ).fit(matrix, target)


def leave_one_session_out_reference(
    train_records: list[dict[str, Any]], user: str, seed: int
) -> np.ndarray:
    """Score each genuine training session with a model that did not fit on it."""
    scores = []
    own_records = [record for record in train_records if record["user_id"] == user]
    for index, held_out in enumerate(own_records):
        model = _fit_user_action_model(
            train_records,
            user,
            seed + index,
            excluded_session=held_out["session_id"],
        )
        actions = held_out["actions"]
        probability = model.predict_proba(actions[ACTION_FEATURES])[:, 1]
        scores.append(1.0 - float(np.mean(probability)))
    return np.asarray(scores, dtype=float)


def time_holdout_reference(
    model: ExtraTreesClassifier, train_records: list[dict[str, Any]], user: str
) -> np.ndarray:
    """Score each later-in-time hold-out from a user's training sessions."""
    scores = []
    for record in train_records:
        if record["user_id"] != user:
            continue
        actions = record["holdout_actions"]
        if actions.empty:
            continue
        probability = model.predict_proba(actions[ACTION_FEATURES])[:, 1]
        scores.append(1.0 - float(np.mean(probability)))
    if not scores:
        raise ValueError(f"No time-hold-out actions available for {user}")
    return np.asarray(scores, dtype=float)


def time_holdout_action_scores(
    model: ExtraTreesClassifier, train_records: list[dict[str, Any]], user: str
) -> np.ndarray:
    """Action-level anomaly scores on a user's later-in-time training hold-out."""
    pieces = [
        1.0 - model.predict_proba(record["holdout_actions"][ACTION_FEATURES])[:, 1]
        for record in train_records
        if record["user_id"] == user and not record["holdout_actions"].empty
    ]
    if not pieces:
        raise ValueError(f"No time-hold-out actions available for {user}")
    return np.concatenate(pieces).astype(float)


def eer_vs_action_count(
    sessions: pd.DataFrame, action_counts: tuple[int | str, ...] = (1, 5, 10, 20, 50, "all")
) -> list[dict[str, Any]]:
    """Measure session EER after averaging the first k action anomaly scores."""
    output = []
    labels = sessions["label"].to_numpy(dtype=int)
    for count in action_counts:
        values = []
        for action_scores in sessions["_action_anomaly_scores"]:
            scores = np.asarray(action_scores, dtype=float)
            limit = len(scores) if count == "all" else min(int(count), len(scores))
            values.append(float(np.mean(scores[:limit])) if limit else 0.5)
        values_array = np.asarray(values, dtype=float)
        eer, threshold, far, frr = equal_error_rate(
            values_array[labels == 0], values_array[labels == 1]
        )
        output.append(
            {
                "actions": count,
                "eer": eer,
                "threshold": threshold,
                "far": far,
                "frr": frr,
            }
        )
    return output


def train_mouse_models(
    data_dir: Path,
    artifact_dir: Path,
    seed: int = 42,
    max_actions_per_session: int = 250,
    time_fraction: float = 0.7,
    test_later_start_fraction: float = 0.3,
) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    """Fit per-user action classifiers and score only public-labeled sessions."""
    labels = pd.read_csv(data_dir / "public_labels.csv").set_index("filename")
    train_paths = sorted((data_dir / "training_files").glob("user*/*"))
    test_paths = sorted((data_dir / "test_files").glob("user*/*"))
    labeled_paths = [path for path in test_paths if path.name in labels.index]
    users = sorted({path.parent.name for path in train_paths})

    train_records, aggregate_rows = load_mouse_training_records(
        train_paths, seed, max_actions_per_session, time_fraction
    )
    aggregate_frame = pd.DataFrame(aggregate_rows)
    deviation_profiles = {
        user: fit_deviation_profile(aggregate_frame[aggregate_frame["user_id"] == user])
        for user in users
    }

    models: dict[str, ExtraTreesClassifier] = {}
    references: dict[str, np.ndarray] = {}
    reference_action_scores: dict[str, np.ndarray] = {}
    thresholds: dict[str, float] = {}
    reference_features: dict[str, list[dict[str, float]]] = {}
    for user_index, user in enumerate(users):
        model = _fit_user_action_model(train_records, user, seed + user_index)
        models[user] = model
        references[user] = time_holdout_reference(model, train_records, user)
        reference_action_scores[user] = time_holdout_action_scores(
            model, train_records, user
        )
        thresholds[user] = float(np.quantile(references[user], 0.95))
        own_records = [
            record
            for record in train_records
            if record["user_id"] == user and not record["holdout_actions"].empty
        ]
        reference_features[user] = [
            {
                "behavioral_anomaly": float(score),
                **session_deviation_features(
                    aggregate_session(record["holdout_actions"]),
                    deviation_profiles[user],
                ),
            }
            for score, record in zip(references[user], own_records)
        ]

    scored_rows: list[dict[str, Any]] = []
    for path in labeled_paths:
        user = path.parent.name
        actions = _read_actions(path)
        _, later_actions, later_split = time_split_actions(
            actions, test_later_start_fraction
        )
        if actions.empty:
            anomaly = 0.5
        else:
            genuine_probability = models[user].predict_proba(actions[ACTION_FEATURES])[:, 1]
            action_anomaly_scores = 1.0 - genuine_probability
            anomaly = float(np.mean(action_anomaly_scores))
        if actions.empty:
            action_anomaly_scores = np.asarray([], dtype=float)
        if later_actions.empty:
            later_action_anomaly_scores = np.asarray([], dtype=float)
            later_anomaly = 0.5
        else:
            later_probability = models[user].predict_proba(
                later_actions[ACTION_FEATURES]
            )[:, 1]
            later_action_anomaly_scores = 1.0 - later_probability
            later_anomaly = float(np.mean(later_action_anomaly_scores))
        session_aggregate = aggregate_session(actions)
        label = int(labels.loc[path.name, "is_illegal"])
        scored_rows.append(
            {
                "user_id": user,
                "session_id": path.name,
                "label": label,
                "behavioral_anomaly": anomaly,
                "later_behavioral_anomaly": later_anomaly,
                "action_count": int(len(actions)),
                "later_action_count": int(len(later_actions)),
                "_action_anomaly_scores": action_anomaly_scores.tolist(),
                "_later_action_anomaly_scores": later_action_anomaly_scores.tolist(),
                "later_dropped_straddlers": later_split["dropped_straddlers"],
                **session_deviation_features(
                    session_aggregate, deviation_profiles[user]
                ),
            }
        )
        aggregate_rows.append(
            {
                "partition": "labeled_test",
                "user_id": user,
                "session_id": path.name,
                "label": label,
                **session_aggregate,
            }
        )

    scored = pd.DataFrame(scored_rows)
    per_user: list[dict[str, Any]] = []
    for user in users:
        subset = scored[scored["user_id"] == user]
        genuine = subset.loc[subset["label"] == 0, "behavioral_anomaly"].to_numpy()
        impostor = subset.loc[subset["label"] == 1, "behavioral_anomaly"].to_numpy()
        eer, threshold, _, _ = equal_error_rate(genuine, impostor)
        fixed_threshold = thresholds[user]
        fixed_far, fixed_frr = far_frr_at_threshold(
            genuine, impostor, fixed_threshold
        )
        per_user.append(
            {
                "user_id": user,
                "n_sessions": int(len(subset)),
                "n_genuine": int((subset["label"] == 0).sum()),
                "n_impostor": int((subset["label"] == 1).sum()),
                "auc": float(roc_auc_score(subset["label"], subset["behavioral_anomaly"])),
                "eer": eer,
                "eer_threshold": threshold,
                "train_time_holdout_p95_threshold": fixed_threshold,
                "far_at_train_time_holdout_p95": fixed_far,
                "frr_at_train_time_holdout_p95": fixed_frr,
            }
        )
    global_eer, global_threshold, _, _ = equal_error_rate(
        scored.loc[scored["label"] == 0, "behavioral_anomaly"].to_numpy(),
        scored.loc[scored["label"] == 1, "behavioral_anomaly"].to_numpy(),
    )
    row_thresholds = scored["user_id"].map(thresholds).to_numpy(dtype=float)
    labels_array = scored["label"].to_numpy(dtype=int)
    scores_array = scored["behavioral_anomaly"].to_numpy(dtype=float)
    fixed_far = float(np.mean(scores_array[labels_array == 1] <= row_thresholds[labels_array == 1]))
    fixed_frr = float(np.mean(scores_array[labels_array == 0] > row_thresholds[labels_array == 0]))
    later_scores = scored["later_behavioral_anomaly"].to_numpy(dtype=float)
    later_genuine = later_scores[labels_array == 0]
    later_impostor = later_scores[labels_array == 1]
    later_eer, later_eer_threshold, _, _ = equal_error_rate(
        later_genuine, later_impostor
    )
    later_fixed_far = float(
        np.mean(later_impostor <= row_thresholds[labels_array == 1])
    )
    later_fixed_frr = float(
        np.mean(later_genuine > row_thresholds[labels_array == 0])
    )
    per_user_split = {}
    for user in users:
        own_records = [record for record in train_records if record["user_id"] == user]
        split_rows = [record["time_split"] for record in own_records]
        starts = [
            row["max_enrollment_action_start_fraction"]
            for row in split_rows
            if row["max_enrollment_action_start_fraction"] is not None
        ]
        per_user_split[user] = {
            "training_sessions": len(own_records),
            "enrollment_actions": int(
                sum(row["enrollment_actions"] for row in split_rows)
            ),
            "holdout_actions": int(sum(row["holdout_actions"] for row in split_rows)),
            "dropped_straddlers": int(
                sum(row["dropped_straddlers"] for row in split_rows)
            ),
            "max_enrollment_action_start_fraction": max(starts) if starts else None,
        }
    all_split_rows = [record["time_split"] for record in train_records]
    max_start_fractions = [
        row["max_enrollment_action_start_fraction"]
        for row in all_split_rows
        if row["max_enrollment_action_start_fraction"] is not None
    ]
    results = {
        "global_auc": float(roc_auc_score(scored["label"], scored["behavioral_anomaly"])),
        "global_eer": global_eer,
        "global_eer_threshold": global_threshold,
        "mean_user_auc": float(np.mean([row["auc"] for row in per_user])),
        "std_user_auc": float(np.std([row["auc"] for row in per_user])),
        "mean_user_eer": float(np.mean([row["eer"] for row in per_user])),
        "std_user_eer": float(np.std([row["eer"] for row in per_user])),
        "labeled_sessions": int(len(scored)),
        "fixed_train_time_holdout_p95": {
            "far": fixed_far,
            "frr": fixed_frr,
            "mean_user_far": float(
                np.mean([row["far_at_train_time_holdout_p95"] for row in per_user])
            ),
            "mean_user_frr": float(
                np.mean([row["frr_at_train_time_holdout_p95"] for row in per_user])
            ),
        },
        "operating_points": {
            "global_far_1pct": global_far_operating_point(
                scores_array[labels_array == 0], scores_array[labels_array == 1], 0.01
            ),
            "global_far_5pct": global_far_operating_point(
                scores_array[labels_array == 0], scores_array[labels_array == 1], 0.05
            ),
        },
        "later_portion_only": {
            "start_fraction": float(test_later_start_fraction),
            "global_auc": float(roc_auc_score(labels_array, later_scores)),
            "global_eer": later_eer,
            "global_eer_threshold": later_eer_threshold,
            "fixed_train_time_holdout_p95_far": later_fixed_far,
            "fixed_train_time_holdout_p95_frr": later_fixed_frr,
            "actions": int(scored["later_action_count"].sum()),
            "dropped_straddlers": int(scored["later_dropped_straddlers"].sum()),
        },
        "time_split": {
            "method": "within_session_elapsed_time",
            "time_fraction": float(time_fraction),
            "test_later_start_fraction": float(test_later_start_fraction),
            "per_user": per_user_split,
            "leakage_checks": {
                "training_files_only": bool(
                    all(record["source_partition"] == "training_files" for record in train_records)
                ),
                "enrollment_starts_before_cut": bool(
                    all(row["enrollment_starts_before_cut"] for row in all_split_rows)
                ),
                "enrollment_ends_at_or_before_cut": bool(
                    all(row["enrollment_ends_at_or_before_cut"] for row in all_split_rows)
                ),
                "max_enrollment_action_start_fraction_below_cut": bool(
                    max(max_start_fractions, default=float("-inf")) < time_fraction
                ),
            },
            "max_enrollment_action_start_fraction": max(
                max_start_fractions, default=None
            ),
        },
        "eer_vs_actions": eer_vs_action_count(scored),
    }

    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "feature_names": ACTION_FEATURES,
            "models": models,
            "reference_scores": references,
            "reference_action_scores": reference_action_scores,
            "thresholds": thresholds,
            "deviation_profiles": deviation_profiles,
            "reference_features": reference_features,
            "reference_source": "training_session_time_holdout",
        },
        artifact_dir / "mouse_models.joblib",
    )
    (artifact_dir / "mouse_per_user.json").write_text(
        json.dumps(per_user, indent=2), encoding="utf-8"
    )
    return results, per_user, scored, pd.DataFrame(aggregate_rows)
