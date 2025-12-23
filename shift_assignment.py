import json
from typing import Optional

import pandas as pd

from transfer_plans import (
    adapt_target_plan_to_frames,
    adapt_past_plans_to_frames,
    collect_history_stats,
    build_training_data,
    train_and_calibrate_with_val,
    assign_target_period,
    get_feature_importance_dict,
)
from visualize_output_plan import ensure_dir, render_visualizations


def run_pipeline(
    job_json_path: str,
    fairness_weekly_cap_hours: Optional[float] = None,  # kept for backward compatibility (maps to soft cap if provided)
    visualization_mode: bool = False,
    output_dir: str = "viz_out",
    fairness_weekly_soft_cap_hours: Optional[float] = 40.0,   # NEW preferred soft cap
    fairness_opt_out_hard_cap_delta_hours: float = 10.0,       # NEW hard cap delta
):
    with open(job_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Historical
    hist_shifts_df, assignments_df = adapt_past_plans_to_frames(data)
    if hist_shifts_df.empty or assignments_df.empty:
        raise ValueError("No historical data found. Ensure past_shift_plans contain shifts and shift_assignments.")

    # Target
    target_shifts, shift_index, users_by_id, customer_tz = adapt_target_plan_to_frames(data)
    if not target_shifts:
        raise ValueError("No target shifts found in shift_plan.shifts.")

    users_df = pd.DataFrame(list(users_by_id.values()))

    # Stats and training
    stats_by_user_ctx = collect_history_stats(hist_shifts_df, assignments_df, users_df, lam=0.85)
    train_df = build_training_data(hist_shifts_df, assignments_df, users_df, stats_by_user_ctx, k_neg_per_pos=5)
    model, iso_cal, feats, (X_val, y_val) = train_and_calibrate_with_val(train_df)

    # Map deprecated fairness_weekly_cap_hours -> soft cap if provided
    soft_cap = fairness_weekly_soft_cap_hours
    if fairness_weekly_cap_hours is not None:
        soft_cap = fairness_weekly_cap_hours

    # Assign target
    result, report = assign_target_period(
        target_shifts, users_by_id, shift_index,
        model, iso_cal, feats, stats_by_user_ctx,            # FIX: use iso_cal and feats
        fairness_weekly_soft_cap_hours=soft_cap,             # soft weekly (FTE-scaled)
        fairness_opt_out_hard_cap_delta_hours=fairness_opt_out_hard_cap_delta_hours,  # hard cap = soft + delta
        customer_tz=customer_tz,
        top_k=5
    )

    report["modelSummary"] = {"featureImportance": get_feature_importance_dict(model, feats)}

    print("Hist shifts:", hist_shifts_df.shape, "Assignments:", assignments_df.shape)
    print("Distinct shiftIds with assignments:", assignments_df["shiftId"].nunique())
    print("Missing shiftIds in history:", len(set(assignments_df["shiftId"]) - set(hist_shifts_df["id"])))
    print(hist_shifts_df[["unit", "shiftType", "weekday"]].drop_duplicates().shape)

    # Write back assignments into shift_plan.shift_assignments
    target_shift_ids = {int(s["id"]) for s in data["shift_plan"].get("shifts", [])}
    existing = data["shift_plan"].get("shift_assignments") or []
    kept = [a for a in existing if int(a["shift_id"]) not in target_shift_ids]
    for sid, uid in result.items():
        kept.append({"shift_id": int(sid), "employee_uuid": uid, "source": "MODEL"})
    data["shift_plan"]["shift_assignments"] = kept

    # Visualization
    if visualization_mode:
        ensure_dir(output_dir)
        viz = render_visualizations(
            model, iso_cal, feats, X_val, y_val, report,
            target_shifts_count=len(target_shifts),
            output_dir=output_dir, sample_topk_shifts=6
        )
        data["reportHtml"] = viz["html"]
        print(f"Visualization written to: {viz['html']}")

    # Attach report
    data["assignmentReport"] = report
    return data


if __name__ == "__main__":
    output = run_pipeline(
        job_json_path="input_files/planning_request_complete_salzhofen.json",
        fairness_weekly_cap_hours=50.0,  # interpreted as soft cap unless fairness_weekly_soft_cap_hours is set explicitly
        visualization_mode=True,
        output_dir="output/viz_out",
        fairness_weekly_soft_cap_hours=40.0,                 # optional explicit soft cap
        fairness_opt_out_hard_cap_delta_hours=10.0,            # hard cap is soft + 10h
    )
    with open("output/output_job.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    assigned_pairs = [
        (a["shift_id"], a["employee_uuid"])
        for a in output["shift_plan"]["shift_assignments"]
        if a.get("source") == "MODEL"
    ]
    print(f"Assigned {len(assigned_pairs)} shifts by model, e.g.:", assigned_pairs[:10])
