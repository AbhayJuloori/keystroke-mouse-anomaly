# Experiment log

The project went through three iterations. Each section lists what changed and what the numbers did, including changes that made things worse.

## First version

### Pipeline compatibility check

The first end-to-end run stopped when MLflow rejected file-backed tracking without its compatibility opt-in. The pipeline set `MLFLOW_ALLOW_FILE_STORE=true` and restarted. No metrics from the stopped attempt were promoted.

### Protocol implementation

Implemented the CMU `sessionIndex` split and asserted 200 enrollment, 200 genuine test, and 250 impostor samples for every subject. The check passed for all 51 subjects.

### Keystroke baselines

Ran scaled Manhattan, Isolation Forest, one-class SVM, and a PyTorch autoencoder with settings fixed in code. Isolation Forest produced the lowest single-detector mean EER at 0.0858. No detector setting was selected from test EER.

### Mouse representation

Used pause and click action segmentation with kinematic features. The detector fits action-level user-versus-other classifiers because each user has only 5 to 7 genuine training sessions. It produced global AUC 0.9307 and mean per-user AUC 0.9412.

### Initial risk fusion

The first version combined the behavioral anomaly scalar with SYNTHETIC device flags in logistic regression. Its limitations motivated the user-relative deviation features and candidate comparison below.

### Initial serving and monitoring

The first version saved per-user detectors, served both modalities through FastAPI, and implemented process-local PSI monitoring.

## Second iteration: specific reasons and fixed thresholds

### User-relative mouse evidence

Added eight signed session deviations against each claimed user's training-session median and MAD scale. The batch path and `/score/session` call the same helper. The values flow into grouped risk evaluation, the saved serving model, and feature-specific SHAP reason codes.

### Risk candidate comparison

Compared fixed logistic regression and fixed gradient-boosted trees with identical five-fold GroupKFold splits. Logistic AUC was 0.9222; tree AUC was 0.9161. Logistic regression was selected by grouped out-of-fold AUC. Gradient-boosted trees did worse on that selection metric and was not used for serving. No candidate settings were tuned against final test metrics.

### Leave-one-session-out references

Replaced in-sample mouse training scores with leave-one-training-session-out genuine scores. Each held-out session is scored by a refit that did not see it. Those scores now define the per-user p95 threshold and the saved drift reference.

### Fixed operating points

Added CMU FAR and FRR at enrollment p90, p95, and p99. Added mouse FAR and FRR at each user's leave-one-session-out p95 threshold. Added risk FAR and FRR at global 1% and 5% FAR thresholds chosen from user-grouped out-of-fold scores. MLflow logs the same numeric fields.

### Added evidence

Added normalized four-detector keystroke fusion, mouse EER after 1, 5, 10, 20, 50, and all actions, a TestClient drift replay, 200-request endpoint latency measurements, Docker packaging, and focused unit tests. Added `mouse_eer_vs_actions.png` and `drift_demo.png`.

### What the Balabit split can show

CMU chronology is verified from `sessionIndex`. Balabit has only within-session elapsed timestamps, so the docs now state that session-to-session order cannot be checked. Mouse fitting uses only the challenge's `training_files`; labeled `test_files` are scored afterward.

## Third iteration: a time split for mouse data and a drift fix

### Within-session time split for Balabit

The second iteration left the Balabit side without a chronological split. `kma.splits.time_split_actions` now cuts every training session at 0.7000 of its elapsed time. Actions that start before the cut and end at or before it are enrollment; actions that start at or after it are the later hold-out; actions that straddle the cut are dropped. Per-user models and deviation baselines fit only on enrollment actions. The later hold-out sets each user's p95 threshold and drift reference. Test sessions are never used for fitting. All leakage checks in `time_split.balabit.leakage_checks` are computed by the code and all came out true.

Before (second iteration, challenge split only, leave-one-session-out thresholds): mouse global AUC 0.9312, mean user AUC 0.9407, mean user EER 0.1202, fixed p95 FAR 0.1506 and FRR 0.1144. After: global AUC 0.9307, mean user AUC 0.9412, mean user EER 0.1163, fixed p95 FAR 0.0667 and FRR 0.3358. The models now see 70% of each training session, so some change is expected. Leave-one-session-out was dropped from serving.

### Drift check fix

The second iteration flagged drift on genuine keystroke traffic after 20 requests (PSI 7.93). Causes: the reference was in-sample enrollment scores, which sit lower than any unseen data; 20 scores into 10 bins left most bins empty; and the mouse reference had only five to seven session scores per user. Fixes: keystroke reference from leave-one-session-out folds within sessions 1 to 4; reference quantile bins with open ends; additive smoothing; minimum windows before any verdict; mouse drift moved to per-action scores against the later-in-time hold-out. A first rerun still flagged 83% of genuine keystroke checkpoints, because a 50-score window has an expected no-change PSI of about 0.22, above the threshold. The keystroke minimum window is now derived from that noise formula (164 scores), and the remaining genuine drift is measured as template aging across sessions 5 to 8. The replay now covers all 51 CMU users and all 10 Balabit users. PSI 0.2 stayed fixed as the rule; 0.1 and 0.25 are reported for context only.

### MLflow read-back

The pipeline now reads its own runs back with `mlflow.search_runs` and fails if any fixed-threshold FAR or FRR key is missing. Verified keys are listed in `mlflow_metric_verification`.

### Data and interpretation limits

The device flags remain synthetic. The final risk model uses public Balabit labels after grouped evaluation. The 51 CMU users and 10 Balabit users do not support demographic fairness claims or production identity decisions.
