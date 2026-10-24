import numpy as np
import pandas as pd

from kma.keystroke import ScaledManhattan, normalized_score_fusion, protocol_split


def test_scaled_manhattan_toy_values():
    model = ScaledManhattan().fit(np.array([[0.0, 0.0], [2.0, 4.0]]))
    np.testing.assert_allclose(model.score_samples([[1.0, 2.0]]), [0.0])
    expected = np.mean([1.0 / np.sqrt(2.0), 2.0 / np.sqrt(8.0)])
    np.testing.assert_allclose(model.score_samples([[2.0, 4.0]]), [expected])


def test_protocol_split_has_time_separation_and_250_impostors():
    rows = []
    for subject_index in range(51):
        subject = f"s{subject_index:03d}"
        for session in range(1, 9):
            for rep in range(1, 51):
                rows.append(
                    {
                        "subject": subject,
                        "sessionIndex": session,
                        "rep": rep,
                        "feature": subject_index + session / 10,
                    }
                )
    data = pd.DataFrame(rows)
    train, genuine, impostor = protocol_split(data, "s000")
    assert len(train) == 200
    assert len(genuine) == 200
    assert len(impostor) == 250
    assert set(train["sessionIndex"]) == {1, 2, 3, 4}
    assert set(genuine["sessionIndex"]) == {5, 6, 7, 8}
    assert not train["sessionIndex"].between(5, 8).any()
    assert train["subject"].eq("s000").all()
    assert impostor["subject"].nunique() == 50
    assert impostor.groupby("subject").size().eq(5).all()


def test_fusion_normalizes_each_detector_from_training_scores():
    training = {
        "small": np.array([0.0, 1.0, 2.0]),
        "large": np.array([100.0, 200.0, 300.0]),
    }
    scores = {
        "small": np.array([1.0, 2.0]),
        "large": np.array([200.0, 300.0]),
    }
    np.testing.assert_allclose(
        normalized_score_fusion(training, scores), np.array([0.0, 1.0])
    )
