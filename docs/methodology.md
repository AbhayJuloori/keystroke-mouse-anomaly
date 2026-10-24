# Methodology

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

Fixed logistic regression and fixed gradient-boosted trees use the same five GroupKFold splits by claimed user. The selected candidate is the one with higher out-of-fold AUC on all inputs. The selected candidate was Logistic regression. It is then evaluated on three predefined stages: behavioral anomaly only; behavior plus deviation features; and behavior plus deviations plus SYNTHETIC flags.

Global risk thresholds for FAR targets of 1% and 5% are the largest observed impostor out-of-fold score that does not exceed each target FAR. Their measured FAR and FRR are stored under `risk.operating_points`. The final serving model is fit on all labeled sessions only after grouped evaluation.

SHAP explains the selected serving model. Linear SHAP is used for logistic regression and Tree SHAP for gradient boosting. Reason text uses both the selected feature and the sign of the observed user-relative deviation, such as faster versus slower or longer versus shorter.

## Synthetic device signals

`configs/device_signals.yaml` fixes label-conditional Bernoulli rates and seed 2026. Every generated column starts with `synthetic_`. These signals test the fusion path. They are not measurements from Balabit or a device.

## Evidence by action count

For each labeled mouse test session, the action-level anomaly probabilities remain in event order. Session scores are recomputed from the first 1, 5, 10, 20, 50, or all available actions. The global EER for each amount of evidence is stored in `mouse.eer_vs_actions`.

## Serving, latency, and monitoring

FastAPI loads saved models, per-user deviation profiles, fixed thresholds, and reference scores. The reported p50 and p95 latencies use 200 in-process TestClient requests per endpoint with real artifacts. They include request validation, feature extraction, scoring, and SHAP reason generation, but not network transit.

The drift demo replays one CMU user's 200 later genuine samples followed by 250 impostor samples through `/score/keystroke`. PSI is recorded after each block of 20 requests. The API retains the latest 200 scores, defines bins from the training reference distribution, and flags drift after at least 10 scores when PSI exceeds 0.2.
