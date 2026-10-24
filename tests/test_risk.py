from pathlib import Path

import numpy as np
import pandas as pd

from kma.mouse import DEVIATION_FEATURES
from kma.risk import ALL_FEATURES, add_synthetic_device_signals, reason_codes


def test_reason_codes_name_feature_and_observed_direction():
    values = pd.Series(0.0, index=ALL_FEATURES)
    values["deviation_speed_mean"] = 2.0
    values["deviation_click_duration"] = -1.0
    values["synthetic_emulator"] = 1.0
    shap_row = np.zeros(len(ALL_FEATURES))
    shap_row[ALL_FEATURES.index("deviation_speed_mean")] = 3.0
    shap_row[ALL_FEATURES.index("deviation_click_duration")] = 2.0
    shap_row[ALL_FEATURES.index("synthetic_emulator")] = 1.0
    reasons = reason_codes(shap_row, values)
    assert "mouse speed much faster than this user's history" in reasons
    assert "click hold time shorter than this user's history" in reasons
    assert "emulator detected (synthetic)" in reasons


def test_synthetic_device_generator_matches_configured_rates():
    sample_size = 100_000
    sessions = pd.DataFrame(
        {
            "label": np.repeat([0, 1], sample_size // 2),
            **{feature: np.zeros(sample_size) for feature in DEVIATION_FEATURES},
        }
    )
    config_path = Path(__file__).parents[1] / "configs" / "device_signals.yaml"
    generated, config = add_synthetic_device_signals(sessions, config_path)
    for label, group in ((0, "genuine"), (1, "impostor")):
        subset = generated[generated["label"] == label]
        for signal, expected in config["rates"][group].items():
            actual = subset[f"synthetic_{signal}"].mean()
            assert abs(actual - expected) < 0.01
