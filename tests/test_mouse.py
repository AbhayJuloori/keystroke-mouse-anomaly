from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import kma.mouse as mouse_module
from kma.mouse import (
    ACTION_FEATURES,
    DEVIATION_SOURCES,
    eer_vs_action_count,
    extract_action_features,
    fit_deviation_profile,
    load_mouse_training_records,
    segment_actions,
    session_deviation_features,
)
from kma.splits import ACTION_TIME_COLUMNS


def test_action_segmentation_splits_click_and_long_pause():
    events = pd.DataFrame(
        {
            "record timestamp": [0.0, 0.1, 0.2, 0.3, 0.4, 12.0, 12.1, 12.2, 12.3],
            "client timestamp": [0.0] * 9,
            "button": ["NoButton", "NoButton", "Left", "Left", "Left"] + ["NoButton"] * 4,
            "state": ["Move", "Move", "Pressed", "Move", "Released"] + ["Move"] * 4,
            "x": [0, 1, 2, 3, 3, 10, 11, 12, 13],
            "y": [0, 0, 1, 2, 2, 5, 5, 5, 5],
        }
    )
    actions = segment_actions(events)
    assert [kind for kind, _ in actions] == ["point_click", "mouse_move"]
    features = extract_action_features(events)
    assert len(features) == 2
    assert features.loc[0, "is_click"] == 1.0
    assert features.loc[1, "is_move"] == 1.0
    assert features["straightness"].between(0, 1).all()


def test_deviation_features_use_training_median_and_signed_direction():
    sources = set(DEVIATION_SOURCES.values())
    training = pd.DataFrame(
        {source: [1.0, 2.0, 3.0] for source in sources}
    )
    profile = fit_deviation_profile(training)
    high = session_deviation_features({source: 4.0 for source in sources}, profile)
    low = session_deviation_features({source: 0.0 for source in sources}, profile)
    assert all(value > 0 for value in high.values())
    assert all(value < 0 for value in low.values())


def test_training_record_loader_rejects_test_file_paths(monkeypatch, tmp_path):
    empty = pd.DataFrame(columns=ACTION_FEATURES + ACTION_TIME_COLUMNS)
    monkeypatch.setattr(mouse_module, "_read_actions", lambda path: empty)
    training_path = tmp_path / "training_files" / "user1" / "session1"
    records, _ = load_mouse_training_records([training_path])
    assert records[0]["path"] == training_path
    test_path = tmp_path / "test_files" / "user1" / "session2"
    with pytest.raises(ValueError, match="not a training file"):
        load_mouse_training_records([test_path])


def test_mouse_fitting_receives_only_training_enrollment_actions(monkeypatch, tmp_path):
    rows = []
    for start, end, marker in [(1.0, 2.0, 1.0), (6.5, 7.5, 2.0), (8.0, 9.0, 3.0)]:
        row = {feature: marker for feature in ACTION_FEATURES}
        row.update(
            {
                "action_start_time": start,
                "action_end_time": end,
                "session_start_time": 0.0,
                "session_end_time": 10.0,
            }
        )
        rows.append(row)
    actions = pd.DataFrame(rows)
    monkeypatch.setattr(mouse_module, "_read_actions", lambda path: actions)
    paths = [
        tmp_path / "training_files" / "user1" / "session1",
        tmp_path / "training_files" / "user2" / "session1",
    ]
    records, _ = load_mouse_training_records(paths, time_fraction=0.7)
    fitted = {}

    class FakeClassifier:
        def __init__(self, **kwargs):
            pass

        def fit(self, matrix, target):
            fitted["matrix"] = matrix.copy()
            fitted["target"] = target.copy()
            return self

    monkeypatch.setattr(mouse_module, "ExtraTreesClassifier", FakeClassifier)
    mouse_module._fit_user_action_model(records, "user1", 42)
    assert set(fitted["matrix"]["duration"]) == {1.0}
    assert all(record["source_partition"] == "training_files" for record in records)
    assert all(record["time_split"]["enrollment_starts_before_cut"] for record in records)


def test_eer_vs_action_count_uses_first_k_scores():
    sessions = pd.DataFrame(
        {
            "label": [0, 0, 1, 1],
            "_action_anomaly_scores": [
                [0.1, 0.1],
                [0.2, 0.2],
                [0.8, 0.8],
                [0.9, 0.9],
            ],
        }
    )
    rows = eer_vs_action_count(sessions, (1, "all"))
    assert [row["actions"] for row in rows] == [1, "all"]
    assert [row["eer"] for row in rows] == [0.0, 0.0]
