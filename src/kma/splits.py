"""Shared chronological split helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd


ACTION_START = "action_start_time"
ACTION_END = "action_end_time"
SESSION_START = "session_start_time"
SESSION_END = "session_end_time"
ACTION_TIME_COLUMNS = [ACTION_START, ACTION_END, SESSION_START, SESSION_END]


def time_split_actions(
    actions: pd.DataFrame, fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Split actions by elapsed session time and drop cut-straddling actions."""
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be between zero and one")
    missing = [column for column in ACTION_TIME_COLUMNS if column not in actions]
    if missing:
        raise ValueError(f"Actions are missing time columns: {missing}")
    if actions.empty:
        return actions.copy(), actions.copy(), {
            "fraction": float(fraction),
            "enrollment_actions": 0,
            "holdout_actions": 0,
            "dropped_straddlers": 0,
            "max_enrollment_action_start_fraction": None,
            "enrollment_starts_before_cut": True,
            "enrollment_ends_at_or_before_cut": True,
        }

    session_start = float(actions[SESSION_START].min())
    session_end = float(actions[SESSION_END].max())
    elapsed = session_end - session_start
    if elapsed <= 0.0:
        raise ValueError("Session elapsed time must be positive")
    cut = session_start + fraction * elapsed

    starts = actions[ACTION_START].astype(float)
    ends = actions[ACTION_END].astype(float)
    straddles = (starts < cut) & (ends > cut)
    enrollment_mask = (starts < cut) & (ends <= cut)
    holdout_mask = starts >= cut
    enrollment = actions.loc[enrollment_mask].copy().reset_index(drop=True)
    holdout = actions.loc[holdout_mask].copy().reset_index(drop=True)
    max_start_fraction = (
        float(((enrollment[ACTION_START].max() - session_start) / elapsed))
        if not enrollment.empty
        else None
    )
    return enrollment, holdout, {
        "fraction": float(fraction),
        "session_start": session_start,
        "session_end": session_end,
        "cut_time": float(cut),
        "enrollment_actions": int(len(enrollment)),
        "holdout_actions": int(len(holdout)),
        "dropped_straddlers": int(straddles.sum()),
        "max_enrollment_action_start_fraction": max_start_fraction,
        "enrollment_starts_before_cut": bool((enrollment[ACTION_START] < cut).all()),
        "enrollment_ends_at_or_before_cut": bool((enrollment[ACTION_END] <= cut).all()),
    }
