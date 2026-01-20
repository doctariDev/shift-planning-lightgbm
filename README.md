# Machine Learning for Shift Assignments

## Overview

We learn assignment patterns from historical rosters and apply them to a target planning period:
1. Data adaptation: Convert past shift plans and assignments into structured training frames.
2. Feature engineering: Build context features per shift and per user, including recency and periodicity.
3. Model training: Train a LightGBM binary classifier to estimate “user will be assigned to this shift”.
4. Calibration: Apply isotonic regression to map raw scores to better-calibrated probabilities.
5. Scoring and selection: For each target shift, filter feasible candidates (hard constraints), score them, blend with fairness/experience signals (soft constraints), and assign the best candidate.
6. Reporting: Produce a detailed assignment report and a compact JSON with ranked candidates per shift.

## Inputs and Artifacts

- Input JSON: `planning_request_complete.json`
  - Contains:
    - `shift_plan`: target shifts and employees
    - `past_shift_plans`: historical shifts and assignments
    - `public_holidays` (optional)
- Notebook/script:
  - `shift_assignment.ipynb` (interactive)
  - `shift_assignment.py` (batch/run)
- Outputs:
  - Detailed report (per shift decision path, candidates, fairness stats)
  - Final-only JSON with top candidates per shift (for downstream merging)

## Data Adaptation

Two adapters prepare the data:
- `adapt_past_plans_to_frames(data)`
  - Produces:
    - `hist_shifts_df`: metadata per historical shift (unit, shiftType, weekday, date, start/end, holiday flag, required qualifications)
    - `assignments_df`: historical links (shiftId, userId)
- `adapt_target_plan_to_frames(data)`
  - Produces:
    - `target_shifts`: the shifts to fill (with meta and `preplannedUserId` if any)
    - `shift_index`: dict by shift id
    - `users_by_id`: available employees in target period
    - `customer_tz`: timezone for temporal parsing

Timezone handling and holiday marking ensure correct local dates, weekday, and `isHoliday`.

## Feature Engineering

We model the probability that a given user will be assigned to a given shift.

- Shift/context features:
  - `unit` (categorical), `shiftType` (categorical), `weekday` (0–6)
  - `hour` (start hour), `duration` (hours), `isHoliday` (0/1)
- User–context historical stats (recency-weighted):
  - `rw_assign_rate`, `count_assigned`, `last_assigned_days`, `rw_assign_rate_holiday`
- Periodicity and recency:
  - `weeks_since_last_ctx_wd`, `worked_last_1w_ctx_wd`, `worked_last_2w_ctx_wd`, `freq_ctx_wd_8w`
- User FTE proxy:
  - `userFTE = weekly_hours / 40.0` from `timed_properties` (if present)

Features are computed consistently for training and inference.

## Training

- Samples:
  - Positives: historical assignments (`y=1`)
  - Negatives: compatible but not assigned users for that shift (`y=0`, typically `k_neg_per_pos=5`)
- Model: LightGBM (objective `binary`)
  - Categorical features: `unit`, `shiftType`
  - Split: stratified 75/25 train/validation
- Calibration: Isotonic regression on validation raw scores → calibrated probabilities

Calibration makes predicted probabilities match observed frequencies.

## Validation Metrics: AUC, AP, Brier

- AUC (Area Under ROC Curve)
  - Measures ranking quality (how often positives outrank negatives across thresholds).
  - 0.5 random, 1.0 perfect. Higher is better.
  - Practical: True assignees appear near the top of ranked candidates.

- AP (Average Precision, area under Precision–Recall)
  - Emphasizes positive class performance under imbalance.
  - Baseline ≈ positive rate; 1.0 perfect. Higher is better.
  - Practical: How concentrated true assignees are among your top-ranked candidates.

- Brier Score (on calibrated probabilities)
  - Mean squared error between predicted probabilities and outcomes.
  - 0.0 perfect. Lower is better.
  - Practical: Trustworthiness of probabilities (e.g., 0.7 ≈ 70% realized).

Interpretation:
- High AUC + High AP → strong ranking; top suggestions are usually right.
- Low Brier → reliable probabilities for downstream fairness/thresholding.
- Good AUC but poor Brier → ranking fine, probabilities miscalibrated (calibration helps).
- Okay AUC but low AP → many near-ties; improve negatives or features.

## Scoring and Assignment

For each target shift:

1. Candidate generation (hard filters)
   - `user_qualified` (required qualifications)
   - `user_available_for_shift` (hook; intended hard availability)
   - `conflicts_with_parallel` (no overlapping assignments in current plan)
   - `negative_wish` (hook for explicit opt-out)
   - Preplanned assignment (`preplannedUserId`) is honored and skipped from scoring

2. Model scoring
   - `score_candidates_for_shift` builds features
   - LightGBM predicts; isotonic calibration produces `p_calibrated`

3. Blended score with soft signals
   - `blended_score = p_calibrated`
     + `0.10 * rw_assign_rate`
     + `0.05 * min(count_assigned/5, 1)`
     + `0.05 * (1 − min(last_days, 60)/60)`
     + `+ 0.08 * rw_assign_rate_holiday` if holiday

4. Fairness soft cap and opt-out hard cap
   - Soft cap penalty via `fairness_penalty(new_hours, soft_cap, hard_cap)`
     - Below soft cap: small penalty (~≤0.05)
     - Between soft and hard cap: graduated penalty (~0.20–0.50)
   - Hard cap: if `new_week_hours` exceeds user hard cap → candidate skipped

5. Selection
   - Choose max `final_score = blended_score − fairness_penalty`
   - Track hours per user per ISO week and globally

6. Reporting
   - Per shift: decision path, filter reasons, candidate list, penalties, chosen user
   - Per user: total assigned hours and count

## Fairness and Constraints

- Hard constraints (exclusion if violated):
  - Qualifications
  - No overlapping assignments within the plan
  - Employee weekly hard cap (`max_weekly_hours` or `soft_cap + delta`)
  - Preserve preplanned assignments
  - Availability

- Soft constraints (affect ranking):
  - ML probability (calibrated)
  - Historical affinity (contextual rates, counts, recency)
  - Holiday experience bonus
  - Weekly soft cap penalty

This separation ensures feasibility first, then merit-based and fair ranking.

## How to Train and Predict Shift Assignments

You can use either the notebook or the Python script:

- Notebook: `shift_assignment.ipynb`
  - Run the cells to:
    - Load `planning_request_complete.json`
    - Adapt data, train model, calibrate, score target shifts
    - Generate report and candidate JSON

- Script: `shift_assignment.py`
  - Run the main method with:
    - Input: `planning_request_complete.json`
    - The script trains on `past_shift_plans` and predicts for the current `shift_plan` period

The pipeline will output:
- A validation printout: `[Validation] AUC=… AP=… Brier=…`
- A detailed assignment report with decision traces
- A compact “final-only” JSON: `build_assigned_output_final_only(report)`

Optionally, use `assign_top_candidates_to_shifts` to merge model suggestions into the base plan:
- Preserves existing assignments; fills only missing shift assignments from the model’s top candidates.

## FAQ

- Why use both AUC and AP?
  - AUC measures global ranking quality across thresholds; AP focuses on precision for the positive class, which is more informative when positives are rare.

- Do calibrated probabilities affect selection?
  - Yes. Calibration improves the trustworthiness of `p_calibrated`, which is the base of `blended_score` and interacts with fairness penalties.

- Can the model override existing assignments?
  - No. Preplanned or existing assignments are preserved and skipped from scoring to respect operational decisions.

- What if no candidates pass the hard cap?
  - The shift will remain unassigned in the model output with notes explaining the hard-cap skips. Adjust caps or consider more candidates.