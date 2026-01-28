import json
import pandas as pd
from utils import (
    adapt_past_plans_to_frames,
    adapt_target_plan_to_frames,
    build_training_data,
    calibrate_isotonic,
    collect_history_stats,
    train_lgb_full,
    save_model_bundle,
    load_model_bundle,
    continue_training as utils_continue_training,
    recalibrate,
    calculate_assignment_scores,
)
from visualize_output_plan import ensure_dir

def train_model(
    planning_request_complete_path: str,
    from_scratch: bool = True,
    recency_weighting: float = 0.85,
    model_bundle_dir: str | None = None,
    num_additional_rounds: int = 200,
    save_updated_bundle: bool = True,
):
    with open(planning_request_complete_path, encoding="utf-8") as f:
        data = json.load(f)

    hist_shifts_df, assignments_df = adapt_past_plans_to_frames(data)
    if hist_shifts_df.empty or assignments_df.empty:
        raise ValueError(
            "No historical data found. Ensure past_shift_plans contain shifts and shift_assignments."
        )

    target_shifts, _, users_by_id, _ = adapt_target_plan_to_frames(data)
    if not target_shifts:
        raise ValueError("No target shifts found in shift_plan.shifts.")

    users_df = pd.DataFrame(list(users_by_id.values()))
    stats_by_user_ctx = collect_history_stats(
        hist_shifts_df, assignments_df, lam=recency_weighting
    )
    train_df = build_training_data(
        hist_shifts_df, assignments_df, users_df, stats_by_user_ctx, k_neg_per_pos=5
    )

    if from_scratch:
        model, feats, (X_val, y_val) = train_lgb_full(train_df)
        iso_cal, _ = calibrate_isotonic(model, X_val, y_val)

        if save_updated_bundle:
            save_model_bundle(model, iso_cal, feats, save_dir=model_bundle_dir or "model_bundle")

        return model, iso_cal, feats, stats_by_user_ctx

    if not model_bundle_dir:
        model_bundle_dir = "model_bundle"
    
    model, iso_cal, feats = load_model_bundle(save_dir=model_bundle_dir)

    cat_cols = feats if feats else ["unit_tags", "workplace_id", "shiftType"]
    for c in cat_cols:
        if train_df[c].dtype.name != "category":
            train_df[c] = train_df[c].astype("category")

    model = utils_continue_training(
        booster=model,
        df=train_df,
        num_additional_rounds=num_additional_rounds,
    )
    iso_cal = recalibrate(model, train_df)

    if save_updated_bundle:
        save_model_bundle(model, iso_cal, feats, save_dir=model_bundle_dir)

    return model, iso_cal, feats, stats_by_user_ctx


def evaluate_model(
    model,
    iso_cal,
    feats,
    planning_request_complete_path: str,
    output_dir: str = "output/assignment_scores.json",
):
    with open(planning_request_complete_path, encoding="utf-8") as f:
        data = json.load(f)

    target_shifts, shift_index, users_by_id, customer_tz = adapt_target_plan_to_frames(data)

    hist_shifts_df, assignments_df = adapt_past_plans_to_frames(data)

    stats_by_user_ctx = collect_history_stats(hist_shifts_df, assignments_df)

    scores = calculate_assignment_scores(
        target_shifts=target_shifts,
        users_by_id=users_by_id,
        shift_index=shift_index,
        model=model,
        iso_calibrator=iso_cal,
        features=feats,
        stats_by_user_ctx=stats_by_user_ctx,
        customer_tz=customer_tz,
        top_k=5
    )

    ensure_dir("output")
    with open(output_dir, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)
    print(f"Wrote output with scores to: {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to planning_request_complete JSON")
    parser.add_argument("--from_scratch", action="store_true", help="Train from scratch")
    parser.add_argument("--bundle_dir", default="model_bundle", help="Model bundle directory")
    parser.add_argument("--rounds", type=int, default=200, help="Additional rounds when continuing")
    parser.add_argument("--no_save", action="store_true", help="Do not save updated bundle")
    args = parser.parse_args()

    model, iso, feats, stats = train_model(
        planning_request_complete_path=args.input,
        from_scratch=args.from_scratch,
        model_bundle_dir=args.bundle_dir,
        num_additional_rounds=args.rounds,
        save_updated_bundle=not args.no_save
    )
    print("Training finished. Features:", feats)
