import json
import pandas as pd

from utils import (
    adapt_target_plan_to_frames,
    adapt_past_plans_to_frames,
    build_assigned_output,
    collect_history_stats,
    build_training_data,
    train_and_calibrate_with_val,
    assign_target_period,
    get_feature_importance_dict,
)
from visualize_output_plan import ensure_dir


def build_assignment_output(report: dict) -> dict:

    assigned = []
    for s in report.get("shifts", []):
        chosen = (s.get("chosen") or {}).get("userId")
        if not chosen:
            continue
        cands = s.get("candidatesAfterFinalScoreOnly") or []
        cands = [
            {"userId": c["userId"], "final_score": float(c.get("final_score", 0.0))}
            for c in cands
        ]
        cands.sort(key=lambda d: d["final_score"], reverse=True)
        assigned.append({
            "shiftId": int(s.get("shiftId")),
            "candidates": cands
        })
    return {"shifts": assigned}


def run_pipeline(
    planning_request_complete_path: str,
    output_dir: str = "output",
):
    with open(planning_request_complete_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    hist_shifts_df, assignments_df = adapt_past_plans_to_frames(data)
    if hist_shifts_df.empty or assignments_df.empty:
        raise ValueError("No historical data found. Ensure past_shift_plans contain shifts and shift_assignments.")

    target_shifts, shift_index, users_by_id, customer_tz = adapt_target_plan_to_frames(data)
    if not target_shifts:
        raise ValueError("No target shifts found in shift_plan.shifts.")

    users_df = pd.DataFrame(list(users_by_id.values()))

    stats_by_user_ctx = collect_history_stats(hist_shifts_df, assignments_df, users_df, lam=0.85)
    train_df = build_training_data(hist_shifts_df, assignments_df, users_df, stats_by_user_ctx, k_neg_per_pos=5)
    model, iso_cal, feats, (X_val, y_val) = train_and_calibrate_with_val(train_df)

    result, report = assign_target_period(
        target_shifts, users_by_id, shift_index,
        model, iso_cal, feats, stats_by_user_ctx,
        customer_tz=customer_tz,
        top_k=5
    )

    report["modelSummary"] = {"featureImportance": get_feature_importance_dict(model, feats)}

    print("Hist shifts:", hist_shifts_df.shape, "Assignments:", assignments_df.shape)
    print("Distinct shiftIds with assignments:", assignments_df["shiftId"].nunique())
    print("Missing shiftIds in history:", len(set(assignments_df["shiftId"]) - set(hist_shifts_df["id"])))
    print(hist_shifts_df[["unit", "shiftType", "weekday"]].drop_duplicates().shape)

    target_shift_ids = {int(s["id"]) for s in data["shift_plan"].get("shifts", [])}
    existing = data["shift_plan"].get("shift_assignments") or []
    kept = [a for a in existing if int(a["shift_id"]) not in target_shift_ids]
    for sid, uid in result.items():
        kept.append({"shift_id": int(sid), "employee_uuid": uid, "source": "MODEL"})
    data["shift_plan"]["shift_assignments"] = kept

    data["assignmentReport"] = report

    compact = build_assigned_output(report)

    ensure_dir("output")
    with open(output_dir, "w", encoding="utf-8") as f:
        json.dump(compact, f, indent=2, ensure_ascii=False)
    print("Wrote output with scores to: output/output_job.json")

    return data