import pandas as pd

from kma.splits import time_split_actions


def _actions():
    return pd.DataFrame(
        {
            "action_start_time": [0.0, 6.5, 8.0],
            "action_end_time": [1.0, 7.5, 9.0],
            "session_start_time": [0.0, 0.0, 0.0],
            "session_end_time": [10.0, 10.0, 10.0],
            "value": [1, 2, 3],
        }
    )


def test_time_split_enrollment_actions_start_before_cut():
    enrollment, holdout, details = time_split_actions(_actions(), 0.7)
    assert enrollment["action_start_time"].lt(details["cut_time"]).all()
    assert holdout["action_start_time"].ge(details["cut_time"]).all()


def test_time_split_drops_actions_that_straddle_cut():
    enrollment, holdout, details = time_split_actions(_actions(), 0.7)
    assert enrollment["value"].tolist() == [1]
    assert holdout["value"].tolist() == [3]
    assert details["dropped_straddlers"] == 1
