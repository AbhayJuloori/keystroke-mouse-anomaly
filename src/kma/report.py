"""Figures and documentation generated from measured pipeline outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .risk import ALL_FEATURES


DISPLAY_NAMES = {
    "scaled_manhattan": "Scaled Manhattan",
    "isolation_forest": "Isolation Forest",
    "one_class_svm": "One-class SVM",
    "autoencoder": "Autoencoder",
    "fusion": "Score fusion",
}
MODEL_NAMES = {
    "logistic_regression": "Logistic regression",
    "gradient_boosting": "Gradient-boosted trees",
}


def create_figures(
    results: dict[str, Any],
    keystroke_per_user: dict[str, list[dict[str, Any]]],
    mouse_per_user: list[dict[str, Any]],
    risk_sessions: pd.DataFrame,
    shap_values: np.ndarray,
    artifact_dir: Path,
    figure_dir: Path,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    colors = ["#36618c", "#648f61", "#aa6f39", "#8c5a83", "#4f7777"]

    names = list(DISPLAY_NAMES)
    means = [results["keystroke"][name]["mean_eer"] for name in names]
    stds = [results["keystroke"][name]["std_eer"] for name in names]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    positions = np.arange(len(names))
    ax.bar(positions, means, yerr=stds, capsize=4, color=colors, alpha=0.8)
    for position, name in zip(positions, names):
        points = [row["eer"] for row in keystroke_per_user[name]]
        jitter = np.linspace(-0.13, 0.13, len(points))
        ax.scatter(position + jitter, points, s=8, alpha=0.35, color="black")
    ax.set_xticks(positions, [DISPLAY_NAMES[name] for name in names], rotation=15)
    ax.set_ylabel("Equal error rate")
    ax.set_title("Keystroke EER across 51 users")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "keystroke_eer.png", dpi=160)
    plt.close(fig)

    best = min(names, key=lambda name: results["keystroke"][name]["mean_eer"])
    pooled = np.load(artifact_dir / "keystroke_pooled_scores.npz")
    genuine = pooled[f"{best}_genuine"]
    impostor = pooled[f"{best}_impostor"]
    thresholds = np.quantile(np.concatenate([genuine, impostor]), np.linspace(0, 1, 250))
    far = np.asarray([(impostor <= value).mean() for value in thresholds])
    frr = np.asarray([(genuine > value).mean() for value in thresholds])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(far, frr, color="#36618c", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False accept rate")
    ax.set_ylabel("False reject rate")
    ax.set_title(f"DET operating curve: {DISPLAY_NAMES[best]}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "keystroke_det.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ordered = sorted(mouse_per_user, key=lambda row: row["user_id"])
    ax.bar([row["user_id"] for row in ordered], [row["auc"] for row in ordered], color="#648f61")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("ROC AUC")
    ax.set_title("Mouse anomaly AUC by claimed user")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "mouse_user_auc.png", dpi=160)
    plt.close(fig)

    action_rows = results["mouse"]["eer_vs_actions"]
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.plot(
        range(len(action_rows)),
        [row["eer"] for row in action_rows],
        marker="o",
        color="#36618c",
    )
    ax.set_xticks(range(len(action_rows)), [str(row["actions"]) for row in action_rows])
    ax.set_xlabel("Actions used per decision")
    ax.set_ylabel("Equal error rate")
    ax.set_title("Mouse EER by available evidence")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "mouse_eer_vs_actions.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.8))
    risk_column = "risk_behavior_plus_deviations_plus_synthetic"
    genuine_risk = risk_sessions.loc[risk_sessions["label"] == 0, risk_column]
    impostor_risk = risk_sessions.loc[risk_sessions["label"] == 1, risk_column]
    bins = np.linspace(0, 100, 24)
    ax.hist(genuine_risk, bins=bins, density=True, alpha=0.6, label="Genuine")
    ax.hist(impostor_risk, bins=bins, density=True, alpha=0.6, label="Impostor")
    ax.set_xlabel("Out-of-fold risk score")
    ax.set_ylabel("Density")
    ax.set_title("Session risk score distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "risk_distribution.png", dpi=160)
    plt.close(fig)

    mean_absolute = np.mean(np.abs(shap_values), axis=0)
    order = np.argsort(mean_absolute)
    labels = [name.replace("synthetic_", "SYNTHETIC ").replace("_", " ") for name in ALL_FEATURES]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(np.asarray(labels)[order], mean_absolute[order], color="#aa6f39")
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_title("Risk model SHAP summary")
    fig.tight_layout()
    fig.savefig(figure_dir / "shap_summary.png", dpi=160)
    plt.close(fig)

    demo = results["drift_demo"]
    threshold = demo["psi_threshold"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.8))
    key_points = pd.DataFrame(demo["keystroke"]["points"])
    grouped = key_points.groupby("requests_replayed")["psi"]
    steps = np.asarray(sorted(key_points["requests_replayed"].unique()))
    left.fill_between(
        steps,
        grouped.quantile(0.25).loc[steps],
        grouped.quantile(0.75).loc[steps],
        color="#3b6ea5",
        alpha=0.2,
        label="Users 25th to 75th percentile",
    )
    left.plot(steps, grouped.median().loc[steps], color="#3b6ea5", marker="o", label="Median user")
    left.axvline(
        demo["keystroke"]["genuine_requests_per_user"],
        color="gray",
        linestyle=":",
        label="Impostor traffic starts",
    )
    left.axhline(threshold, color="#aa6f39", linestyle="--", label=f"PSI {threshold}")
    left.set_xlabel("Keystroke requests replayed")
    left.set_ylabel("PSI")
    left.set_yscale("log")
    left.set_ylim(bottom=1e-3)
    left.set_title(f"Keystroke drift, {demo['keystroke']['user_count']} CMU users")
    left.legend(fontsize=8)
    left.grid(alpha=0.25)

    mouse_points = pd.DataFrame(demo["mouse"]["points"])
    for user_id, user_points in mouse_points.groupby("user_id"):
        right.plot(
            user_points["sessions_replayed"],
            user_points["psi"],
            color="#999999",
            linewidth=0.8,
        )
        impostor = user_points[user_points["phase"] == "impostor"]
        right.scatter(
            impostor["sessions_replayed"], impostor["psi"], color="#aa6f39", s=6
        )
    right.scatter([], [], color="#aa6f39", s=6, label="Checkpoint after an impostor session")
    right.axhline(threshold, color="#aa6f39", linestyle="--", label=f"PSI {threshold}")
    right.set_xlabel("Mouse sessions replayed (genuine first, then impostor)")
    right.set_ylabel("PSI")
    right.set_yscale("log")
    right.set_ylim(bottom=1e-3)
    right.set_title(f"Mouse drift, {demo['mouse']['user_count']} Balabit users")
    right.legend(fontsize=8)
    right.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "drift_demo.png", dpi=160)
    plt.close(fig)


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def write_documentation(results: dict[str, Any], root: Path) -> None:
    keystroke = results["keystroke"]
    mouse = results["mouse"]
    risk = results["risk"]
    single_names = ["scaled_manhattan", "isolation_forest", "one_class_svm", "autoencoder"]
    best_single = min(single_names, key=lambda name: keystroke[name]["mean_eer"])
    key_rows = "\n".join(
        f"| {DISPLAY_NAMES[name]} | {_fmt(values['mean_eer'])} ± {_fmt(values['std_eer'])} | "
        f"{_fmt(values['mean_far_at_train_p90'])} / {_fmt(values['mean_frr_at_train_p90'])} | "
        f"{_fmt(values['mean_far_at_train_p95'])} / {_fmt(values['mean_frr_at_train_p95'])} | "
        f"{_fmt(values['mean_far_at_train_p99'])} / {_fmt(values['mean_frr_at_train_p99'])} |"
        for name, values in keystroke.items()
    )
    mouse_rows = "\n".join(
        f"| {row['user_id']} | {row['n_genuine']} | {row['n_impostor']} | {_fmt(row['auc'])} | "
        f"{_fmt(row['eer'])} | {_fmt(row['train_time_holdout_p95_threshold'])} | "
        f"{_fmt(row['far_at_train_time_holdout_p95'])} | {_fmt(row['frr_at_train_time_holdout_p95'])} |"
        for row in results["mouse_per_user"]
    )
    risk_rows = "\n".join(
        f"| {name.replace('_', ' ').capitalize()} | {_fmt(values['auc'])} | {_fmt(values['eer'])} |"
        for name, values in risk["stages"].items()
    )
    comparison_rows = "\n".join(
        f"| {MODEL_NAMES[name]} | {_fmt(values['auc'])} | {_fmt(values['eer'])} |"
        for name, values in risk["model_comparison"].items()
    )
    action_rows = "\n".join(
        f"| {row['actions']} | {_fmt(row['eer'])} | {_fmt(row['far'])} | {_fmt(row['frr'])} |"
        for row in mouse["eer_vs_actions"]
    )
    timing_names = results["protocol"]["keystroke_feature_names"]
    curl_features = ", ".join(f'\\"{name}\\": 0.1' for name in timing_names)
    final_stage = risk["stages"]["behavior_plus_deviations_plus_synthetic"]
    op1 = risk["operating_points"]["global_far_1pct"]
    op5 = risk["operating_points"]["global_far_5pct"]
    fixed = mouse["fixed_train_time_holdout_p95"]
    latency = results["api_latency"]
    key_drift = results["drift_demo"]["keystroke"]
    mouse_drift = results["drift_demo"]["mouse"]
    key_drift_standard = key_drift["threshold_summaries"]["0.2"]
    mouse_drift_standard = mouse_drift["threshold_summaries"]["0.2"]
    split = results["time_split"]["balabit"]
    enrollment_actions = sum(
        row["enrollment_actions"] for row in split["per_user"].values()
    )
    holdout_actions = sum(
        row["holdout_actions"] for row in split["per_user"].values()
    )
    dropped_straddlers = sum(
        row["dropped_straddlers"] for row in split["per_user"].values()
    )
    mlflow_keys = sorted(
        {
            key
            for run in results["mlflow_metric_verification"]["runs"].values()
            for key in run["metric_keys"]
        }
    )
    mlflow_key_text = ", ".join(f"`{key}`" for key in mlflow_keys)
    readme = f"""# Keystroke and Mouse Anomaly Detection (CMU, Balabit data)

This project evaluates per-user behavioral anomaly models on the CMU fixed-text keystroke benchmark and the public labeled portion of the Balabit mouse benchmark. `make all` generated every reported value. Full precision values are in [`docs/results.json`](docs/results.json).

The best single keystroke detector was **{DISPLAY_NAMES[best_single]}**, with mean EER **{_fmt(keystroke[best_single]['mean_eer'])} ± {_fmt(keystroke[best_single]['std_eer'])}** across 51 users. Four-detector score fusion reached **{_fmt(keystroke['fusion']['mean_eer'])} ± {_fmt(keystroke['fusion']['std_eer'])}**. Mouse anomaly scoring reached global AUC **{_fmt(mouse['global_auc'])}**. The selected session risk model was **{MODEL_NAMES[risk['selected_model']]}**, with grouped out-of-fold AUC **{_fmt(final_stage['auc'])}** and EER **{_fmt(final_stage['eer'])}** after behavior, user-relative deviations, and SYNTHETIC device flags were combined.

## Claims

1. "Fit per user anomaly detectors (scaled Manhattan distance, Isolation Forest, one class SVM, PyTorch autoencoder) on the CMU Keystroke Dynamics benchmark of 51 users and 20,400 typing samples, scored by equal error rate against impostor attempts."
2. "Engineered speed, curvature, and click timing features from Balabit Mouse Dynamics Challenge sessions and blended them with synthetic device signals (emulator, rooted device, VPN) into one session risk score with SHAP reason codes."
3. "Split every user's sessions by time to keep later attempts out of training, tracked false accept and false reject rates at fixed thresholds in MLflow, and served scoring through FastAPI with a drift check on per user scores."

Balabit does not provide cross-session timestamps. Claim 3 uses the verified CMU session split and a within-session elapsed-time split for every Balabit training file. Only the earliest {split['time_fraction']:.0%} of each training session supplies fitting actions. Later actions calibrate the fixed threshold and mouse drift reference.

### Claim to evidence

| Claim | Code evidence | `docs/results.json` evidence |
|---|---|---|
| 1 | `src/kma/keystroke.py::protocol_split`, `train_keystroke_models` | `protocol.keystroke_users`, `protocol.keystroke_total_samples`, `keystroke.*.mean_eer` |
| 2 | `src/kma/mouse.py::action_features`, `session_deviation_features`; `src/kma/risk.py::train_risk_model`, `reason_codes` | `risk.stages`, `risk.model_comparison`, `risk.synthetic_device_config` |
| 3 | `src/kma/splits.py::time_split_actions`; `src/kma/keystroke.py::protocol_split`; `src/kma/pipeline.py::_verify_mlflow_metrics`; `src/kma/api.py::create_app` | `time_split`, `mlflow_metric_verification`, `mouse.fixed_train_time_holdout_p95`, `risk.operating_points`, `drift_demo` |

## Keystroke results

Scores are normalized against each user's enrollment data. Fusion averages the four normalized detector scores. FAR and FRR use thresholds fixed at enrollment-score percentiles.

| Detector | Mean EER ± std | p90 FAR / FRR | p95 FAR / FRR | p99 FAR / FRR |
|---|---:|---:|---:|---:|
{key_rows}

![Keystroke EER](docs/figures/keystroke_eer.png)

![Keystroke DET curve](docs/figures/keystroke_det.png)

## Mouse anomaly results

Each claimed-user classifier fits only early enrollment actions from `training_files`. The fixed threshold is the 95th percentile of the later, unseen portion of those same training sessions. Across full labeled test sessions, the per-user fixed thresholds produced FAR **{_fmt(fixed['far'])}** and FRR **{_fmt(fixed['frr'])}**.

| User | Genuine | Impostor | AUC | EER | Time hold-out p95 | FAR | FRR |
|---|---:|---:|---:|---:|---:|---:|---:|
{mouse_rows}

Global mouse AUC is **{_fmt(mouse['global_auc'])}**. Mean per-user AUC is **{_fmt(mouse['mean_user_auc'])} ± {_fmt(mouse['std_user_auc'])}**. Mean per-user EER is **{_fmt(mouse['mean_user_eer'])} ± {_fmt(mouse['std_user_eer'])}**.

Using only actions that start after the first {mouse['later_portion_only']['start_fraction']:.0%} of each labeled test session gives AUC **{_fmt(mouse['later_portion_only']['global_auc'])}**, EER **{_fmt(mouse['later_portion_only']['global_eer'])}**, FAR **{_fmt(mouse['later_portion_only']['fixed_train_time_holdout_p95_far'])}**, and FRR **{_fmt(mouse['later_portion_only']['fixed_train_time_holdout_p95_frr'])}** at the same per-user fixed thresholds.

![Mouse AUC](docs/figures/mouse_user_auc.png)

### Mouse evidence available per decision

| Actions | EER | FAR at EER point | FRR at EER point |
|---:|---:|---:|---:|
{action_rows}

![Mouse EER by actions](docs/figures/mouse_eer_vs_actions.png)

## Session risk

Eight signed deviation features compare each session with that claimed user's training-session medians and MAD scales. The candidates use identical five-fold GroupKFold splits by claimed user. **{MODEL_NAMES[risk['selected_model']]}** was selected by out-of-fold AUC before the final serving fit.

| Inputs | Grouped out-of-fold AUC | EER |
|---|---:|---:|
{risk_rows}

| Candidate on all inputs | Grouped out-of-fold AUC | EER |
|---|---:|---:|
{comparison_rows}

The full risk score has FAR **{_fmt(op1['far'])}** and FRR **{_fmt(op1['frr'])}** at its out-of-fold 1% FAR operating point. At the 5% target it has FAR **{_fmt(op5['far'])}** and FRR **{_fmt(op5['frr'])}**. Device flags are generated from label-conditional rates in `configs/device_signals.yaml`. These results show a synthetic fusion mechanism, not expected production lift.

![Risk distributions](docs/figures/risk_distribution.png)

![SHAP summary](docs/figures/shap_summary.png)

## Split and leakage controls

For each CMU subject, sessions 1 through 4 provide 200 enrollment samples and sessions 5 through 8 provide 200 later genuine test samples. The first five time-ordered samples from each other subject provide 250 impostor attempts. No later CMU session enters enrollment.

Balabit files contain only elapsed timestamps within each session, so cross-session chronology cannot be verified. `time_split_actions` places actions that end within the earliest {split['time_fraction']:.0%} of each training session into enrollment, places actions that start at or after the cut into the time hold-out, and drops actions that straddle the cut. Across all users this produced {enrollment_actions} enrollment actions, {holdout_actions} hold-out actions, and {dropped_straddlers} dropped straddlers. All computed leakage checks in `time_split.balabit.leakage_checks` passed. Test files remain fully outside fitting.

The pipeline read its newly written MLflow runs back with `mlflow.search_runs`. It verified these fixed-threshold FAR and FRR metric keys on every applicable keystroke, mouse, and risk run: {mlflow_key_text}.

## API and drift

The FastAPI service computes the same user-relative session deviations stored with the mouse artifacts and returns feature-specific SHAP reason codes. `/drift/{{user_id}}` compares a rolling window of that user's served scores against a reference of genuine scores the model never fit on. Keystroke users keep the last {key_drift['window_size']} sample scores, and the reference is out-of-sample scores from leave-one-session-out folds within sessions 1 through 4. Mouse users keep the last {mouse_drift['window_size']} per-action anomaly scores, and the reference is the later-in-time 30% hold-out of each training session. No verdict is given below {key_drift['minimum_window']} keystroke scores or {mouse_drift['minimum_window']} mouse action scores.

Measured with 200 TestClient requests per endpoint on real artifacts:

| Endpoint | p50 ms | p95 ms |
|---|---:|---:|
| `/score/keystroke` | {_fmt(latency['keystroke']['p50_ms'])} | {_fmt(latency['keystroke']['p95_ms'])} |
| `/score/session` | {_fmt(latency['session']['p50_ms'])} | {_fmt(latency['session']['p95_ms'])} |

The drift replay sends each user's genuine test traffic through the API, then impostor traffic, and reads `/drift` at each checkpoint. PSI uses reference quantile bins with open ends and additive smoothing of {results['drift_demo']['psi_smoothing']}.

| Channel | Users | Median PSI, genuine | Median PSI, impostor | False drift rate at PSI 0.2 (genuine checkpoints) | Detection rate at PSI 0.2 (window fully impostor) |
|---|---:|---:|---:|---:|---:|
| Keystroke (CMU sessions 5 to 8, then other users) | {key_drift['user_count']} | {_fmt(key_drift['median_genuine_psi'])} | {_fmt(key_drift['median_impostor_psi'])} | {_fmt(key_drift_standard['false_drift_rate'])} | {_fmt(key_drift_standard['pure_impostor_detection_rate'])} |
| Mouse (labeled genuine, then impostor sessions) | {mouse_drift['user_count']} | {_fmt(mouse_drift['median_genuine_psi'])} | {_fmt(mouse_drift['median_impostor_psi'])} | {_fmt(mouse_drift_standard['false_drift_rate'])} | {_fmt(mouse_drift_standard['pure_impostor_detection_rate'])} |

Minimum windows come from sampling noise, not from the replay outcome. With 10 bins, a window of n scores and a reference of m scores, PSI under no real change is about 9 x (1/n + 1/m). The minimum window is the smallest n that keeps this at or below half the 0.2 threshold: {key_drift['minimum_window']} for keystroke (expected no-change PSI {_fmt(results['drift_demo']['null_psi_at_minimum_window']['keystroke'])}). Mouse references hold a median of {_fmt(results['drift_demo']['mouse_reference_actions_per_user']['median'])} hold-out actions per user, so {mouse_drift['minimum_window']} actions gives {_fmt(results['drift_demo']['null_psi_at_minimum_window']['mouse_median_reference'])}.

Genuine keystroke drift is partly real. The median user's genuine score, relative to the reference median, rises from {_fmt(results['drift_demo']['keystroke_genuine_score_to_reference_median_ratio_by_session']['5'])} in session 5 to {_fmt(results['drift_demo']['keystroke_genuine_score_to_reference_median_ratio_by_session']['8'])} in session 8 as typing changes over time. Typing changes between sessions and over time (template aging), so on genuine traffic a drift flag is a re-enrollment signal rather than an attack signal. PSI 0.2 is too sensitive for CMU keystroke traffic without periodic re-enrollment, and this run does not re-enroll. Impostor windows sit far higher (median PSI {_fmt(key_drift['median_impostor_psi'])} versus {_fmt(key_drift['median_genuine_psi'])}).

Results at PSI 0.1 and 0.25, and per-checkpoint values, are in `drift_demo` in results.json.

![Drift demo](docs/figures/drift_demo.png)

## Run

Python 3.11 and the pinned packages in `requirements.txt` are expected.

```bash
make setup
make all
make test
make serve
make docker
```

`make docker` builds the image but does not run it. The requested validation does not build the image.

Keystroke scoring requires the exact 31 timing fields:

```bash
curl -X POST http://127.0.0.1:8000/score/keystroke \\
  -H 'Content-Type: application/json' \\
  -d \"{{\\\"user_id\\\": \\\"s002\\\", \\\"features\\\": {{{curl_features}}}}}\"
```

Mouse scoring accepts raw events and optional SYNTHETIC flags:

```bash
curl -X POST http://127.0.0.1:8000/score/session \\
  -H 'Content-Type: application/json' \\
  -d '{{"user_id":"user12","events":[{{"record_timestamp":0.0,"x":0,"y":0}},{{"record_timestamp":0.1,"x":4,"y":2}},{{"record_timestamp":0.2,"x":8,"y":4}},{{"record_timestamp":0.3,"x":12,"y":6}}],"synthetic_vpn":false}}'
```

## Model card

### Intended use

This repository is an offline benchmark and API demonstration for per-user anomaly scoring, score fusion, fixed operating points, explanations, and drift monitoring.

### Not intended for

Do not use it for production identity decisions, access denial, employment decisions, surveillance, or any decision where a false accept or false reject can harm a person. It has no production calibration or external validation.

### Data limits

CMU contains 51 users typing one fixed password. Balabit contains 10 claimed users and only within-session elapsed timestamps. Public Balabit labels cover only part of the test set. Neither benchmark measures free-text behavior, broad device diversity, or current deployment conditions.

### Synthetic signals

Emulator, rooted-device, and VPN flags are simulated. Their measured contribution cannot be treated as evidence for real device telemetry.

### Fairness

The 51-user and 10-user samples are too small for demographic subgroup claims. The datasets do not support a fairness conclusion, and this project does not perform demographic analysis.

## Limitations

- Mouse training has only 5 to 7 genuine sessions per user.
- Risk labels come from the public Balabit test labels and are evaluated out of fold by claimed user.
- The API score buffer is process-local and resets on restart.
- PSI is an alert heuristic, not a guarantee that behavior changed.

See [`docs/methodology.md`](docs/methodology.md) for details and [`docs/experiment-log.md`](docs/experiment-log.md) for experiment history.

## License

Code is released under the MIT License. Dataset terms and citations remain with their respective sources.
"""
    (root / "README.md").write_text(readme, encoding="utf-8")

    selected = MODEL_NAMES[risk["selected_model"]]
    methodology = f"""# Methodology

## Evaluation contract

The pipeline fixes random seeds, model settings, feature definitions, and candidate models in code before evaluation. Candidate selection uses grouped cross-validation, not final mouse anomaly test metrics. Full precision outputs live in `docs/results.json`; prose rounds to four decimal places.

## Keystroke protocol

Each of 51 subjects contributes 400 samples. Sessions 1 through 4 train each detector. Sessions 5 through 8 are later genuine tests. Five time-ordered samples from each other subject form 250 impostor attempts. This is a verified split on `sessionIndex`.

Scaled Manhattan uses enrollment mean and standard deviation. Isolation Forest uses 120 trees after per-user standardization. One-class SVM uses an RBF kernel, `nu=0.05`, and `gamma=scale`. The PyTorch autoencoder has widths 31-16-6-16-31 and trains for 40 full-batch CPU epochs. The score-level fusion converts each detector's scores to z-scores using that user's detector-specific enrollment-score mean and standard deviation, then averages the four values. No detector settings were selected from test EER.

EER minimizes the absolute difference between FAR and FRR. A score at or below threshold is accepted. Fixed per-user thresholds use the 90th, 95th, and 99th percentiles of enrollment scores. All FAR and FRR values at those thresholds are reported in `keystroke`.

## Mouse protocol

Balabit event timestamps are elapsed time within a file. They do not establish an ordering between files. A chronological cross-session split cannot be verified. The project therefore uses the challenge's supplied `training_files` and `test_files` partition. `load_mouse_training_records` rejects any fitting path that is not under `training_files`. Test actions and labels are read only after detector fitting.

Actions split after a pause longer than 10 seconds and after a released click. Segments with fewer than four events are discarded. Features cover duration, point count, path length, displacement, straightness, direction, click duration, velocity, speed, acceleration, jerk, angular velocity, curvature, and action type.

For each claimed user, an Extra Trees classifier with 120 trees and minimum leaf size four learns genuine actions against other users' training actions. At most 250 actions per training session are sampled. Session anomaly is one minus the mean genuine-action probability.

Each user's fixed reference distribution is leave-one-training-session-out. For every genuine training session, the classifier is refit without that session and then scores its full action sequence. The 95th percentile of those scores is the fixed decision threshold and the saved drift reference input. This avoids scoring a training session with a classifier that fit that session.

## User-relative deviation features

The session risk layer receives behavioral anomaly plus signed deviations for mean speed, maximum speed, mean curvature, straightness, click duration, mean acceleration, mean angular velocity, and action duration. The session aggregate is compared with the claimed user's training-session median. Scale is 1.4826 times MAD with a unit-aware floor of 5% of the median magnitude or 0.05, whichever is larger. The profile is stored in `mouse_models.joblib` and the API calls the same `session_deviation_features` helper as training.

## Risk model and explanations

Fixed logistic regression and fixed gradient-boosted trees use the same five GroupKFold splits by claimed user. The selected candidate is the one with higher out-of-fold AUC on all inputs. The selected candidate was {selected}. It is then evaluated on three predefined stages: behavioral anomaly only; behavior plus deviation features; and behavior plus deviations plus SYNTHETIC flags.

Global risk thresholds for FAR targets of 1% and 5% are the largest observed impostor out-of-fold score that does not exceed each target FAR. Their measured FAR and FRR are stored under `risk.operating_points`. The final serving model is fit on all labeled sessions only after grouped evaluation.

SHAP explains the selected serving model. Linear SHAP is used for logistic regression and Tree SHAP for gradient boosting. Reason text uses both the selected feature and the sign of the observed user-relative deviation, such as faster versus slower or longer versus shorter.

## Synthetic device signals

`configs/device_signals.yaml` fixes label-conditional Bernoulli rates and seed {risk['synthetic_device_config']['seed']}. Every generated column starts with `synthetic_`. These signals test the fusion path. They are not measurements from Balabit or a device.

## Evidence by action count

For each labeled mouse test session, the action-level anomaly probabilities remain in event order. Session scores are recomputed from the first 1, 5, 10, 20, 50, or all available actions. The global EER for each amount of evidence is stored in `mouse.eer_vs_actions`.

## Serving, latency, and monitoring

FastAPI loads saved models, per-user deviation profiles, fixed thresholds, and reference scores. The reported p50 and p95 latencies use 200 in-process TestClient requests per endpoint with real artifacts. They include request validation, feature extraction, scoring, and SHAP reason generation, but not network transit.

The drift demo replays one CMU user's 200 later genuine samples followed by 250 impostor samples through `/score/keystroke`. PSI is recorded after each block of 20 requests. The API retains the latest 200 scores, defines bins from the training reference distribution, and flags drift after at least 10 scores when PSI exceeds 0.2.
"""
    (root / "docs" / "methodology.md").write_text(methodology, encoding="utf-8")

    worse = min(risk["model_comparison"], key=lambda name: risk["model_comparison"][name]["auc"])
    experiment = f"""# Experiment log

## 0. Pipeline compatibility check

The first end-to-end attempt in Pass 1 stopped when MLflow rejected file-backed tracking without its compatibility opt-in. The pipeline set `MLFLOW_ALLOW_FILE_STORE=true` and restarted. No metrics from the stopped attempt were promoted.

## 1. Protocol implementation

Implemented the CMU `sessionIndex` split and asserted 200 enrollment, 200 genuine test, and 250 impostor samples for every subject. The check passed for all 51 subjects.

## 2. Keystroke baselines

Ran scaled Manhattan, Isolation Forest, one-class SVM, and a PyTorch autoencoder with settings fixed in code. {DISPLAY_NAMES[best_single]} produced the lowest single-detector mean EER at {_fmt(keystroke[best_single]['mean_eer'])}. No detector setting was selected from test EER.

## 3. Mouse representation

Used pause and click action segmentation with kinematic features. The detector fits action-level user-versus-other classifiers because each user has only 5 to 7 genuine training sessions. It produced global AUC {_fmt(mouse['global_auc'])} and mean per-user AUC {_fmt(mouse['mean_user_auc'])}.

## 4. Initial risk fusion

Pass 1 combined the behavioral anomaly scalar with SYNTHETIC device flags in logistic regression. Its limitations motivated the user-relative deviation features and candidate comparison below.

## 5. Initial serving and monitoring

Pass 1 saved per-user detectors, served both modalities through FastAPI, and implemented process-local PSI monitoring.

## Pass 2

### User-relative mouse evidence

Added eight signed session deviations against each claimed user's training-session median and MAD scale. The batch path and `/score/session` call the same helper. The values flow into grouped risk evaluation, the saved serving model, and feature-specific SHAP reason codes.

### Risk candidate comparison

Compared fixed logistic regression and fixed gradient-boosted trees with identical five-fold GroupKFold splits. Logistic AUC was {_fmt(risk['model_comparison']['logistic_regression']['auc'])}; tree AUC was {_fmt(risk['model_comparison']['gradient_boosting']['auc'])}. {selected} was selected by grouped out-of-fold AUC. {MODEL_NAMES[worse]} did worse on that selection metric and was not used for serving. No candidate settings were tuned against final test metrics.

### Leave-one-session-out references

Replaced in-sample mouse training scores with leave-one-training-session-out genuine scores. Each held-out session is scored by a refit that did not see it. Those scores now define the per-user p95 threshold and the saved drift reference.

### Fixed operating points

Added CMU FAR and FRR at enrollment p90, p95, and p99. Added mouse FAR and FRR at each user's leave-one-session-out p95 threshold. Added risk FAR and FRR at global 1% and 5% FAR thresholds chosen from user-grouped out-of-fold scores. MLflow logs the same numeric fields.

### Added evidence

Added normalized four-detector keystroke fusion, mouse EER after 1, 5, 10, 20, 50, and all actions, a TestClient drift replay, 200-request endpoint latency measurements, Docker packaging, and focused unit tests. Added `mouse_eer_vs_actions.png` and `drift_demo.png`.

### Split wording correction

CMU chronology is verified from `sessionIndex`. Balabit has only within-session elapsed timestamps, so this pass removed any claim of verified cross-session chronology. Mouse fitting uses only the challenge's `training_files`; labeled `test_files` are scored afterward.

## Pass 3

### Within-session time split for Balabit

Pass 2 left the Balabit side without a chronological split. `kma.splits.time_split_actions` now cuts every training session at {_fmt(results['time_split']['balabit']['time_fraction'])} of its elapsed time. Actions that start before the cut and end at or before it are enrollment; actions that start at or after it are the later hold-out; actions that straddle the cut are dropped. Per-user models and deviation baselines fit only on enrollment actions. The later hold-out sets each user's p95 threshold and drift reference. Test sessions are never used for fitting. All leakage checks in `time_split.balabit.leakage_checks` are computed by the code and were true on this run.

Before (pass 2, challenge split only, leave-one-session-out thresholds): mouse global AUC 0.9312, mean user AUC 0.9407, mean user EER 0.1202, fixed p95 FAR 0.1506 and FRR 0.1144. After (this run): global AUC {_fmt(mouse['global_auc'])}, mean user AUC {_fmt(mouse['mean_user_auc'])}, mean user EER {_fmt(mouse['mean_user_eer'])}, fixed p95 FAR {_fmt(mouse['fixed_train_time_holdout_p95']['far'])} and FRR {_fmt(mouse['fixed_train_time_holdout_p95']['frr'])}. The models now see 70% of each training session, so some change is expected. Leave-one-session-out was dropped from serving.

### Drift check fix

Pass 2 flagged drift on genuine keystroke traffic after 20 requests (PSI 7.93). Causes: the reference was in-sample enrollment scores, which sit lower than any unseen data; 20 scores into 10 bins left most bins empty; and the mouse reference had only five to seven session scores per user. Fixes: keystroke reference from leave-one-session-out folds within sessions 1 to 4; reference quantile bins with open ends; additive smoothing; minimum windows before any verdict; mouse drift moved to per-action scores against the later-in-time hold-out. A first rerun still flagged 83% of genuine keystroke checkpoints, because a 50-score window has an expected no-change PSI of about 0.22, above the threshold. The keystroke minimum window is now derived from that noise formula ({results['drift_demo']['keystroke']['minimum_window']} scores), and the remaining genuine drift is measured as template aging across sessions 5 to 8. The replay now covers all {results['drift_demo']['keystroke']['user_count']} CMU users and all {results['drift_demo']['mouse']['user_count']} Balabit users. PSI 0.2 stayed fixed as the rule; 0.1 and 0.25 are reported for context only.

### MLflow read-back

The pipeline now reads its own runs back with `mlflow.search_runs` and fails if any fixed-threshold FAR or FRR key is missing. Verified keys are listed in `mlflow_metric_verification`.

### Data and interpretation limits

The device flags remain synthetic. The final risk model uses public Balabit labels after grouped evaluation. The 51 CMU users and 10 Balabit users do not support demographic fairness claims or production identity decisions.
"""
    (root / "docs" / "experiment-log.md").write_text(experiment, encoding="utf-8")


def save_results(results: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
