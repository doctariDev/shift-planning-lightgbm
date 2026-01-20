import json
import os
import warnings
from collections import defaultdict
from datetime import date as date_cls
from datetime import datetime
from datetime import time as time_cls
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# =========================
# Timezone helper
# =========================
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

def dt_parse_iso(s: str, tz: str | None) -> datetime:
    dt = pd.to_datetime(s)
    if tz and ZoneInfo:
        try:
            if dt.tzinfo is None:
                dt = dt.tz_localize("UTC")
            dt = dt.tz_convert(ZoneInfo(tz))
        except Exception:
            try:
                dt = pd.to_datetime(s).to_pydatetime()
            except Exception:
                pass
    return dt.to_pydatetime()

# =========================
# Holidays: day-based matching
# =========================

def parse_holiday_days(holiday_list: list[dict[str, Any]]) -> set:
    days = set()
    for h in holiday_list or []:
        dstr = h.get("date")
        if not dstr:
            continue
        try:
            day = pd.to_datetime(dstr).date()
            days.add(day)
        except Exception:
            continue
    return days

def is_holiday_day(shift_start_dt: datetime, holiday_days: set) -> bool:
    try:
        local_day = shift_start_dt.date()
        return local_day in holiday_days
    except Exception:
        return False

# =========================
# Core utilities
# =========================

def get_feature_importance_dict(model, features):
    imp_gain = model.feature_importance(importance_type="gain")
    return sorted(
        [{"feature": f, "gain": float(g)} for f, g in zip(features, imp_gain, strict=False)],
        key=lambda x: -x["gain"]
    )

def parse_time_simple(tstr: str) -> time_cls:
    parts = list(map(int, tstr.split(":")))
    if len(parts) == 2:
        return time_cls(parts[0], parts[1])
    return time_cls(parts[0], parts[1], parts[2])

def to_date(obj) -> date_cls | None:
    if obj is None:
        return None
    if isinstance(obj, str):
        try:
            return datetime.fromisoformat(obj.replace("Z", "+00:00")).date()
        except Exception:
            return pd.to_datetime(obj).date()
    if isinstance(obj, datetime):
        return obj.date()
    if hasattr(obj, "to_pydatetime"):
        return obj.to_pydatetime().date()
    if hasattr(obj, "date"):
        try:
            return obj.date()
        except Exception:
            pass
    if isinstance(obj, date_cls):
        return obj
    raise TypeError(f"Unsupported date type: {type(obj)}")

def in_date_range(date_obj, start: str | None, end: str | None) -> bool:
    d = to_date(date_obj)
    if d is None:
        return True
    if start:
        if d < to_date(start):
            return False
    if end:
        if d > to_date(end):
            return False
    return True

def iso_week_index_from_date(d: date_cls) -> int:
    y, w, _ = d.isocalendar()
    return int(y) * 100 + int(w)

def rolling_weeks_freq(weeks_sorted: list[int], current_week_idx: int, window: int = 8) -> float:
    if not weeks_sorted:
        return 0.0
    lo = current_week_idx - window
    cnt = sum(1 for wk in weeks_sorted if lo <= wk < current_week_idx)
    return cnt / float(window)

# =========================
# Hard constraints
# =========================

def user_qualified(user: dict[str, Any], required_quals: list[int]) -> bool:
    uq = {q["id"] for q in user.get("qualifications", [])}
    return set(required_quals or []).issubset(uq)

def conflicts_with_parallel(shift: dict[str, Any], user_id: str, assigned_by_user: dict[str, list[int]], shift_index: dict[int, dict[str, Any]]) -> bool:
    for sid in assigned_by_user.get(user_id, []):
        other = shift_index.get(sid)
        if not other:
            continue
        if shift["date"] == other["date"]:
            s_start = parse_time_simple(shift["start"])
            s_end   = parse_time_simple(shift["end"])
            o_start = parse_time_simple(other["start"])
            o_end   = parse_time_simple(other["end"])
            if not (s_end <= o_start or o_end <= s_start):
                return True
    return False

def negative_wish(shift: dict[str, Any], user_id: str) -> bool:
    return False

def positive_wishers(shift: dict[str, Any]) -> list[str]:
    return []

# =========================
# Adapters
# =========================

def _extract_unit_tags(sh: dict[str, Any]) -> str:
    tags = sh.get("unit_tags")
    if tags is None:
        return "UNTAGGED"
    return str(tags)

def adapt_past_plans_to_frames(data: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    past_plans = data.get("past_shift_plans", []) or []
    default_tz = ((data.get("shift_plan") or {}).get("customer") or {}).get("zone_id")

    shift_rows = []
    asg_rows = []

    for p in past_plans:
        tz = (p.get("customer") or {}).get("zone_id") or default_tz
        holiday_days = parse_holiday_days(p.get("public_holidays") or [])

        for sh in p.get("shifts", []):
            sid = int(sh["id"])
            start_dt_local = dt_parse_iso(sh["start_date_time"], tz)
            end_dt_local = dt_parse_iso(sh["end_date_time"], tz)
            is_hol = is_holiday_day(start_dt_local, holiday_days)

            shift_rows.append({
                "id": sid,
                "unit_tags": _extract_unit_tags(sh),
                "workplace_id": str(sh.get("workplace_id")),
                "shiftType": str(sh.get("shift_card_id") or "GEN"),
                "weekday": int(start_dt_local.weekday()),
                "date": start_dt_local.date().isoformat(),
                "start": start_dt_local.strftime("%H:%M"),
                "end": end_dt_local.strftime("%H:%M"),
                "isHoliday": int(is_hol),
                "requiredQualifications": [int(q) for q in sh.get("qualification_ids", [])]
            })
        for a in p.get("shift_assignments", []):
            asg_rows.append({"shiftId": int(a["shift_id"]), "userId": a["employee_uuid"]})

    hist_shifts_df = pd.DataFrame(shift_rows)
    assignments_df = pd.DataFrame(asg_rows)
    return hist_shifts_df, assignments_df

def adapt_target_plan_to_frames(data: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]], dict[str, dict[str, Any]], str | None]:
    sp = data["shift_plan"]
    customer_tz = (sp.get("customer") or {}).get("zone_id")
    holiday_days = parse_holiday_days(sp.get("public_holidays") or [])

    existing_by_shift = {}
    for a in (sp.get("shift_assignments") or []):
        try:
            sid = int(a.get("shift_id"))
            existing_by_shift[sid] = a.get("employee_uuid")
        except (TypeError, ValueError):
            continue

    target_shifts = []
    for sh in sp.get("shifts", []):
        sid = int(sh["id"])
        start_dt_local = dt_parse_iso(sh["start_date_time"], customer_tz)
        end_dt_local = dt_parse_iso(sh["end_date_time"], customer_tz)
        is_hol = is_holiday_day(start_dt_local, holiday_days)
        target_shifts.append({
            "id": sid,
            "unit_tags": _extract_unit_tags(sh),
            "workplace_id": str(sh.get("workplace_id")),
            "shiftType": str(sh.get("shift_card_id") or "GEN"),
            "weekday": int(start_dt_local.weekday()),
            "date": start_dt_local.date().isoformat(),
            "start": start_dt_local.strftime("%H:%M"),
            "end": end_dt_local.strftime("%H:%M"),
            "isHoliday": int(is_hol),
            "requiredQualifications": [int(q) for q in sh.get("qualification_ids", [])],
            "preplannedUserId": existing_by_shift.get(sid),
            "requests": []
        })

    shift_index = {s["id"]: s for s in target_shifts}
    users_by_id = {e["uuid"]: e for e in sp.get("employees", [])}
    return target_shifts, shift_index, users_by_id, customer_tz

# =========================
# Stats, training data
# =========================

def recency_weight(ts: datetime, now: datetime, lam=0.85, unit="week") -> float:
    delta_days = (now - ts).days
    k = max(delta_days / 7.0, 0) if unit == "week" else max(delta_days, 0)
    return lam ** k

def collect_history_stats(hist_shifts_df: pd.DataFrame,
                          assignments_df: pd.DataFrame,
                          users_df: pd.DataFrame,
                          lam: float = 0.85) -> dict[str, dict[tuple, dict[str, float]]]:
    """
    Build recency-weighted stats by (userId, context), where context = (unit_tags, shiftType, weekday).
    Ensures:
      - unit_tags is present and treated as a stable string categorical.
      - Merge uses matching columns.
      - Weeks worked and holiday flags are computed consistently.
    """
    # Early exit if no history
    if hist_shifts_df.empty:
        return {}

    # Ensure required columns exist
    required_cols = ["id", "unit_tags", "shiftType", "weekday", "date", "isHoliday"]
    missing = [c for c in required_cols if c not in hist_shifts_df.columns]
    if missing:
        raise ValueError(f"hist_shifts_df missing required columns for stats: {missing}")

    # Normalize types
    df = hist_shifts_df.copy()
    # Keep unit_tags as provided (string categorical)
    df["unit_tags"] = df["unit_tags"].astype(str)
    df["shiftType"] = df["shiftType"].astype(str)
    df["weekday"] = df["weekday"].astype(int)
    # Parse date to Timestamp
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["isHoliday"] = df["isHoliday"].fillna(0).astype(int)

    # Reference "now" for recency weighting
    if df["date"].notna().any():
        now = df["date"].max()
    else:
        now = pd.Timestamp(datetime.utcnow())
    now_dt = pd.to_datetime(now)

    def _init():
        return {
            "rw_denom": 0.0, "rw_num": 0.0,
            "count_occurrences": 0, "count_assigned": 0,
            "last_assigned_days": 9999.0,
            "rw_denom_holiday": 0.0, "rw_num_holiday": 0.0,
            "count_occurrences_holiday": 0, "count_assigned_holiday": 0,
            "weeks_worked": []
        }

    stats = defaultdict(lambda: defaultdict(_init))

    # Context key
    df["context"] = df.apply(lambda r: (r["unit_tags"], r["shiftType"], r["weekday"]), axis=1)
    context_by_shift = dict(zip(df["id"], df["context"]))

    # If no assignments, return empty stats (initialized)
    if assignments_df.empty:
        return stats

    # Merge assignments with shift metadata
    # Ensure assignments_df has expected columns
    req_assign_cols = ["shiftId", "userId"]
    miss_assign = [c for c in req_assign_cols if c not in assignments_df.columns]
    if miss_assign:
        raise ValueError(f"assignments_df missing required columns: {miss_assign}")

    merged = assignments_df.merge(
        df[["id", "unit_tags", "shiftType", "weekday", "date", "isHoliday"]],
        left_on="shiftId", right_on="id", how="left"
    )

    # Iterate and accumulate stats
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce")

    for _, row in merged.iterrows():
        uid = row.get("userId")
        sid = row.get("shiftId")
        if pd.isna(uid) or pd.isna(sid):
            continue
        # Context lookup; skip if missing (e.g., assignment references unknown shift)
        ctx = context_by_shift.get(sid)
        if ctx is None:
            continue

        date_dt = row.get("date")
        if pd.isna(date_dt):
            continue

        # Recency weight
        w = recency_weight(date_dt.to_pydatetime() if hasattr(date_dt, "to_pydatetime") else date_dt, now_dt, lam=lam, unit="week")
        sh_is_hol = int(row.get("isHoliday", 0)) == 1

        s = stats[uid][ctx]

        # Count occurrences in context (one per assignment row)
        s["rw_num"] += w
        s["rw_denom"] += w
        s["count_assigned"] += 1
        s["count_occurrences"] += 1
        # Days since last assignment in context
        try:
            delta_days = (now_dt - pd.to_datetime(date_dt)).days
        except Exception:
            delta_days = 9999
        s["last_assigned_days"] = min(s["last_assigned_days"], float(delta_days))

        # Week index for periodic features
        try:
            wk_idx = iso_week_index_from_date(pd.to_datetime(date_dt).date())
            s["weeks_worked"].append(wk_idx)
        except Exception:
            pass

        if sh_is_hol:
            s["rw_num_holiday"] += w
            s["rw_denom_holiday"] += w
            s["count_assigned_holiday"] += 1
            s["count_occurrences_holiday"] += 1

    # Finalize derived rates and periodic lists
    for uid, ctxs in stats.items():
        for ctx, s in ctxs.items():
            s["rw_assign_rate"] = (s["rw_num"] / max(s["rw_denom"], 1e-6))
            if s["rw_denom_holiday"] > 0:
                s["rw_assign_rate_holiday"] = s["rw_num_holiday"] / max(s["rw_denom_holiday"], 1e-6)
            else:
                s["rw_assign_rate_holiday"] = 0.0
            # De-duplicate and sort weeks
            s["weeks_worked"] = sorted(set(s["weeks_worked"]))

    return stats


def build_training_data(hist_shifts_df: pd.DataFrame,
                        assignments_df: pd.DataFrame,
                        users_df: pd.DataFrame,
                        stats_by_user_ctx: dict[str, dict[tuple, dict[str, float]]],
                        k_neg_per_pos: int = 5) -> pd.DataFrame:
    if assignments_df.empty:
        raise ValueError("assignments_df is empty; need historical assignments to train.")
    shift_ids = set(assignments_df["shiftId"].unique())
    hist_shifts_df = hist_shifts_df[hist_shifts_df["id"].isin(shift_ids)].copy()
    hist_shifts_df = hist_shifts_df.sort_values(["date", "start"]).drop_duplicates(subset=["id"], keep="first")
    shift_meta = hist_shifts_df.set_index("id").to_dict(orient="index")
    users_by_id = users_df.set_index("uuid").to_dict(orient="index")

    rows = []
    rng = np.random.default_rng(42)

    def periodic_feats(uid: str, ctx: tuple, cur_date_str: str) -> dict:
        sstats = stats_by_user_ctx.get(uid, {}).get(ctx, {})
        weeks = sstats.get("weeks_worked", []) or []
        cur_week_idx = iso_week_index_from_date(to_date(cur_date_str))
        prev_weeks = [w for w in weeks if w < cur_week_idx]
        if prev_weeks:
            last_w = prev_weeks[-1]
            ws = cur_week_idx - last_w
        else:
            ws = 999.0
        return {
            "weeks_since_last_ctx_wd": float(ws),
            "worked_last_1w_ctx_wd": 1.0 if ws == 1 else 0.0,
            "worked_last_2w_ctx_wd": 1.0 if ws == 2 else 0.0,
            "freq_ctx_wd_8w": rolling_weeks_freq(weeks, cur_week_idx, window=8)
        }

    for sid, g in assignments_df.groupby("shiftId"):
        if sid not in shift_meta:
            continue
        sm = shift_meta[sid]
        ctx = (sm["unit_tags"], sm["shiftType"], sm["weekday"])
        start_s = sm["start"]
        end_s = sm["end"]
        try:
            hour_val = int(str(start_s)[:2])
        except Exception:
            hour_val = 8
        try:
            s_date = to_date(sm["date"])
            duration_val = (datetime.combine(s_date, parse_time_simple(end_s)) -
                            datetime.combine(s_date, parse_time_simple(start_s))).seconds / 3600.0
        except Exception:
            duration_val = 8.0

        req = sm.get("requiredQualifications", [])

        compatible = []
        for uid, u in users_by_id.items():
            if not set(req or []).issubset({q["id"] for q in u.get("qualifications", [])}):
                continue
            compatible.append(uid)

        pos_users = list(g["userId"].values)
        for uid in pos_users:
            sstats = stats_by_user_ctx.get(uid, {}).get(ctx, {})
            urec = users_by_id.get(uid, {})
            pfeats = periodic_feats(uid, ctx, sm["date"])
            rows.append({
                "shiftId": sid, "userId": uid, "y": 1,
                "unit_tags": sm["unit_tags"],
                "workplace_id": sm.get("workplace_id"),
                "shiftType": sm["shiftType"], "weekday": sm["weekday"],
                "hour": hour_val, "duration": duration_val, "isHoliday": int(sm.get("isHoliday", 0)),
                "rw_assign_rate": sstats.get("rw_assign_rate", 0.0),
                "count_assigned": sstats.get("count_assigned", 0),
                "last_assigned_days": sstats.get("last_assigned_days", 9999.0),
                "userFTE": (urec.get("timed_properties", [{}])[0].get("weekly_hours", 40.0))/40.0,
                **pfeats
            })
        neg_pool = [u for u in compatible if u not in pos_users]
        if len(neg_pool) > 0:
            sample_size = min(k_neg_per_pos * len(pos_users), len(neg_pool))
            neg_sample = list(rng.choice(neg_pool, size=sample_size, replace=False))
            for uid in neg_sample:
                sstats = stats_by_user_ctx.get(uid, {}).get(ctx, {})
                urec = users_by_id.get(uid, {})
                pfeats = periodic_feats(uid, ctx, sm["date"])
                rows.append({
                    "shiftId": sid, "userId": uid, "y": 0,
                    "unit_tags": sm["unit_tags"],
                    "workplace_id": sm.get("workplace_id"),
                    "shiftType": sm["shiftType"], "weekday": sm["weekday"],
                    "hour": hour_val, "duration": duration_val, "isHoliday": int(sm.get("isHoliday", 0)),
                    "rw_assign_rate": sstats.get("rw_assign_rate", 0.0),
                    "count_assigned": sstats.get("count_assigned", 0),
                    "last_assigned_days": sstats.get("last_assigned_days", 9999.0),
                    "userFTE": (urec.get("timed_properties", [{}])[0].get("weekly_hours", 40.0))/40.0,
                    **pfeats
                })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("Training data ended up empty. Check past_shift_plans.shift_assignments and ids.")
    df["unit_tags"] = df["unit_tags"].astype("category")
    df["shiftType"] = df["shiftType"].astype("category")
    df["workplace_id"] = df["workplace_id"].astype("category")
    return df

# =========================
# Model training
# =========================

def train_and_calibrate_with_val(df: pd.DataFrame) -> tuple[Any, Any, list[str], tuple[pd.DataFrame, pd.Series]]:
    features = [
        "unit_tags", "workplace_id", "shiftType", "weekday", "hour", "duration", "isHoliday",
        "rw_assign_rate", "count_assigned", "last_assigned_days", "userFTE",
        "weeks_since_last_ctx_wd", "worked_last_1w_ctx_wd", "worked_last_2w_ctx_wd", "freq_ctx_wd_8w"
    ]
    X = df[features]
    y = df["y"]
    cat_features = ["unit_tags", "workplace_id", "shiftType"]

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    lgb_train = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_features, free_raw_data=False)
    params = {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": 50,
        "verbosity": -1,
        "seed": 42
    }
    model = lgb.train(params, lgb_train, num_boost_round=500)
    p_raw_val = model.predict(X_val)
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_raw_val, y_val)
    auc_val = roc_auc_score(y_val, p_raw_val)
    ap_val = average_precision_score(y_val, p_raw_val)
    brier = brier_score_loss(y_val, iso.predict(p_raw_val))
    print(f"[Validation] AUC={auc_val:.3f} AP={ap_val:.3f} Brier={brier:.3f}")
    return model, iso, features, (X_val, y_val)

# =========================
# Scoring and assignment
# =========================

def score_candidates_for_shift(shift: dict[str, Any],
                               candidate_ids: list[str],
                               users_by_id: dict[str, dict[str, Any]],
                               stats_by_user_ctx: dict[str, dict[tuple, dict[str, float]]],
                               model, iso_calibrator, features: list[str]) -> list[tuple[str, float]]:
    ctx = (shift["unit_tags"], shift["shiftType"], shift["weekday"])
    try:
        hour = int(str(shift["start"])[:2])
    except Exception:
        hour = 8
    try:
        s_date = to_date(shift["date"])
        duration = (datetime.combine(s_date, parse_time_simple(shift["end"])) -
                    datetime.combine(s_date, parse_time_simple(shift["start"]))).seconds / 3600.0
    except Exception:
        duration = 8.0
    isHoliday = int(shift.get("isHoliday", 0))

    def periodic_feats_infer(uid: str, ctx: tuple, cur_date_str: str) -> dict:
        sstats = stats_by_user_ctx.get(uid, {}).get(ctx, {})
        weeks = sstats.get("weeks_worked", []) or []
        cur_week_idx = iso_week_index_from_date(to_date(cur_date_str))
        prev_weeks = [w for w in weeks if w < cur_week_idx]
        if prev_weeks:
            last_w = prev_weeks[-1]
            ws = cur_week_idx - last_w
        else:
            ws = 999.0
        return {
            "weeks_since_last_ctx_wd": float(ws),
            "worked_last_1w_ctx_wd": 1.0 if ws == 1 else 0.0,
            "worked_last_2w_ctx_wd": 1.0 if ws == 2 else 0.0,
            "freq_ctx_wd_8w": rolling_weeks_freq(weeks, cur_week_idx, window=8)
        }

    rows, idx = [], []
    cur_date_str = shift["date"]
    for uid in candidate_ids:
        s = stats_by_user_ctx.get(uid, {}).get(ctx, {})
        urec = users_by_id.get(uid, {})
        pfeats = periodic_feats_infer(uid, ctx, cur_date_str)
        rows.append({
            "unit_tags": shift["unit_tags"], "workplace_id": shift.get("workplace_id"),
            "shiftType": shift["shiftType"], "weekday": shift["weekday"],
            "hour": hour, "duration": duration, "isHoliday": isHoliday,
            "rw_assign_rate": s.get("rw_assign_rate", 0.0),
            "count_assigned": s.get("count_assigned", 0),
            "last_assigned_days": s.get("last_assigned_days", 9999.0),
            "userFTE": (urec.get("timed_properties", [{}])[0].get("weekly_hours", 40.0))/40.0,
            **pfeats
        })
        idx.append(uid)
    if not rows:
        return []
    X = pd.DataFrame(rows)
    X["unit_tags"] = X["unit_tags"].astype("category")
    X["shiftType"] = X["shiftType"].astype("category")
    X["workplace_id"] = X["workplace_id"].astype("category")
    p_raw = model.predict(X[features])
    p = iso_calibrator.predict(p_raw)
    return list(zip(idx, p))

def assign_target_period(
    target_shifts: list[dict[str, Any]],
    users_by_id: dict[str, dict[str, Any]],
    shift_index: dict[int, dict[str, Any]],
    model, iso_calibrator, features,
    stats_by_user_ctx: dict[str, dict[tuple, dict[str, float]]],
    fairness_weekly_soft_cap_hours: float | None = None,
    fairness_opt_out_hard_cap_delta_hours: float | None = None,
    customer_tz: str | None = None,
    top_k: int = 5
) -> tuple[dict[int, str], dict[str, Any]]:
    assigned_by_user: dict[str, list[int]] = defaultdict(list)
    assigned_hours_by_user_global: dict[str, float] = defaultdict(float)
    assigned_hours_by_user_week: dict[tuple[str, str], float] = defaultdict(float)
    all_assignments_by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)

    result = {}
    report = {
        "shifts": [],
        "usersSummary": {},
        "modelSummary": {},
        "runConfig": {
            "fairness_weekly_soft_cap_hours_default": fairness_weekly_soft_cap_hours,
            "fairness_opt_out_hard_cap_delta_hours_default": fairness_opt_out_hard_cap_delta_hours,
            "top_k": top_k
        }
    }

    def week_key_from_date_str(dstr: str) -> str:
        d = to_date(dstr)
        y, w, _ = d.isocalendar()
        return f"{y}-W{w:02d}"

    def fairness_penalty(new_hours: float, soft_cap: float | None, hard_cap: float | None) -> float:
        if soft_cap is None or soft_cap <= 0:
            return 0.0
        if new_hours <= soft_cap:
            proximity = new_hours / max(soft_cap, 1e-6)
            return 0.05 * proximity
        if hard_cap is not None and new_hours < hard_cap:
            over = new_hours - soft_cap
            span = max(hard_cap - soft_cap, 1e-6)
            severity = min(over / span, 1.0)
            return 0.20 + 0.30 * severity
        return 1.0

    target_shifts = sorted(target_shifts, key=lambda x: (x["date"], x["start"]))

    for s in target_shifts:
        sid = s["id"]

        preplanned_user = s.get("preplannedUserId")
        if preplanned_user:
            result[sid] = preplanned_user
            report["shifts"].append({
                "shiftId": sid,
                "meta": {
                    "unit_tags": s["unit_tags"], "workplace_id": s.get("workplace_id"),
                    "shiftType": s["shiftType"], "weekday": s["weekday"],
                    "date": s["date"], "start": s["start"], "end": s["end"],
                    "requiredQualifications": s.get("requiredQualifications", []),
                    "isHoliday": int(s.get("isHoliday", 0))
                },
                "preplannedUserId": preplanned_user,
                "candidatesBeforeFilters": 0,
                "filteredOut": [],
                "candidatesAfterFilters": [],
                "topCandidates": [],
                "candidatesAfterFinalScoreOnly": [],
                "unconstrainedCandidatesAfterFinalScoreOnly": [],
                "chosen": {
                    "userId": preplanned_user,
                    "score_after_penalty": None,
                    "base_score": None,
                    "fairness_penalty": None,
                    "week_key": None,
                    "week_hours_after": None
                },
                "decisionPath": ["skipped_due_to_existing_assignment"],
                "notes": ["skipped_scoring_due_to_existing_assignment"]
            })
            continue

        explanation = {
            "shiftId": sid,
            "meta": {
                "unit_tags": s["unit_tags"], "workplace_id": s.get("workplace_id"),
                "shiftType": s["shiftType"], "weekday": s["weekday"],
                "date": s["date"], "start": s["start"], "end": s["end"],
                "requiredQualifications": s.get("requiredQualifications", []),
                "isHoliday": int(s.get("isHoliday", 0))
            },
            "preplannedUserId": s.get("preplannedUserId"),
            "positiveWishers": positive_wishers(s),
            "candidatesBeforeFilters": [],
            "filteredOut": [],
            "candidatesAfterFilters": [],
            "topCandidates": [],
            "candidatesAfterFinalScoreOnly": [],
            "unconstrainedCandidatesAfterFinalScoreOnly": [],
            "chosen": None,
            "decisionPath": [],
            "notes": []
        }

        explanation["decisionPath"].append("candidate_generation")
        all_user_ids = list(users_by_id.keys())

        for uid in all_user_ids:
            reasons = []
            user = users_by_id[uid]
            if not user_qualified(user, s.get("requiredQualifications", [])):
                reasons.append("not_qualified")
            # Availability hook, if you have data:
            # if not user_available_for_shift(user, s, customer_tz):
            #     reasons.append("not_available")
            if conflicts_with_parallel(s, uid, assigned_by_user, shift_index):
                reasons.append("parallel_conflict")
            if negative_wish(s, uid):
                reasons.append("negative_wish")
            if reasons:
                explanation["filteredOut"].append({"userId": uid, "reasons": reasons})
            else:
                explanation["candidatesAfterFilters"].append(uid)
        explanation["candidatesBeforeFilters"] = len(all_user_ids)

        unconstrained_candidates = [uid for uid in all_user_ids if uid != preplanned_user]

        if not explanation["candidatesAfterFilters"]:
            explanation["notes"].append("no feasible candidates (constrained)")
            explanation["decisionPath"].append("scoring_unconstrained")
            unconstrained_ranked = score_candidates_for_shift(
                s, unconstrained_candidates, users_by_id, stats_by_user_ctx, model, iso_calibrator, features
            )
            ctx = (s["unit_tags"], s["shiftType"], s["weekday"])
            stats_for_uid_uc = {uid: (stats_by_user_ctx.get(uid, {}) or {}).get(ctx, {}) for uid, _ in unconstrained_ranked}

            beta1, beta2, beta3 = 0.10, 0.05, 0.05
            C, L = 5.0, 60.0
            is_holiday = int(s.get("isHoliday", 0)) == 1
            holiday_weight = 0.08

            def blended_score(uid: str, p: float) -> float:
                st = stats_for_uid_uc.get(uid, {}) or {}
                rw = float(st.get("rw_assign_rate", 0.0))
                rw_hol = float(st.get("rw_assign_rate_holiday", 0.0))
                cnt = float(st.get("count_assigned", 0.0))
                last_days = float(st.get("last_assigned_days", 9999.0))
                cnt_norm = min(cnt / C, 1.0)
                recency_bonus = max(0.0, 1.0 - min(last_days, L) / L)
                holiday_bonus = holiday_weight * rw_hol if is_holiday else 0.0
                return float(p) + beta1 * rw + beta2 * cnt_norm + beta3 * recency_bonus + holiday_bonus

            try:
                hrs = (datetime.combine(to_date(s["date"]), parse_time_simple(s["end"])) -
                       datetime.combine(to_date(s["date"]), parse_time_simple(s["start"]))).seconds / 3600.0
            except Exception:
                hrs = 8.0
            wk = week_key_from_date_str(s["date"])

            for uid, p in unconstrained_ranked:
                urec = users_by_id.get(uid, {})
                tp = (urec.get("timed_properties") or [{}])[0]
                soft_cap_uid = tp.get("weekly_hours", fairness_weekly_soft_cap_hours or 40.0)
                hard_cap_uid = tp.get("max_weekly_hours", None)
                if hard_cap_uid is None and fairness_opt_out_hard_cap_delta_hours is not None:
                    hard_cap_uid = soft_cap_uid + fairness_opt_out_hard_cap_delta_hours

                current_week_hours = assigned_hours_by_user_week[(uid, wk)]
                new_hours = current_week_hours + hrs

                base = blended_score(uid, p)
                pen = fairness_penalty(new_hours, soft_cap_uid, hard_cap_uid)
                final_uc = base - pen
                explanation["unconstrainedCandidatesAfterFinalScoreOnly"].append({
                    "userId": uid,
                    "final_score": float(final_uc)
                })
            explanation["unconstrainedCandidatesAfterFinalScoreOnly"].sort(key=lambda d: d["final_score"], reverse=True)
            report["shifts"].append(explanation)
            continue

        explanation["decisionPath"].append("scoring_constrained")
        ranked = score_candidates_for_shift(
            s, explanation["candidatesAfterFilters"], users_by_id, stats_by_user_ctx, model, iso_calibrator, features
        )

        if not ranked:
            explanation["notes"].append("scoring produced no results (constrained)")
            explanation["decisionPath"].append("scoring_unconstrained")
            unconstrained_ranked = score_candidates_for_shift(
                s, unconstrained_candidates, users_by_id, stats_by_user_ctx, model, iso_calibrator, features
            )
            explanation["unconstrainedCandidatesAfterFinalScoreOnly"] = [
                {"userId": uid, "final_score": float(p)} for uid, p in sorted(unconstrained_ranked, key=lambda t: t[1], reverse=True)
            ]
            report["shifts"].append(explanation)
            continue

        ctx = (s["unit_tags"], s["shiftType"], s["weekday"])
        stats_for_uid = {uid: (stats_by_user_ctx.get(uid, {}) or {}).get(ctx, {}) for uid, _ in ranked}

        top_list_report = sorted(ranked, key=lambda t: t[1], reverse=True)[:top_k]
        enriched_top = []
        for uid, p in top_list_report:
            sstats = stats_for_uid.get(uid, {}) or {}
            urec = users_by_id.get(uid, {})
            enriched_top.append({
                "userId": uid,
                "p_calibrated": float(p),
                "rw_assign_rate": sstats.get("rw_assign_rate", 0.0),
                "rw_assign_rate_holiday": sstats.get("rw_assign_rate_holiday", 0.0),
                "count_assigned": int(sstats.get("count_assigned", 0)),
                "count_assigned_holiday": int(sstats.get("count_assigned_holiday", 0)),
                "last_assigned_days": float(sstats.get("last_assigned_days", 9999.0)),
                "userFTE": (urec.get("timed_properties", [{}])[0].get("weekly_hours", 40.0))/40.0
            })
        explanation["topCandidates"] = enriched_top

        beta1, beta2, beta3 = 0.10, 0.05, 0.05
        C, L = 5.0, 60.0
        is_holiday = int(s.get("isHoliday", 0)) == 1
        holiday_weight = 0.08

        def blended_score(uid: str, p: float) -> float:
            st = stats_for_uid.get(uid, {}) or {}
            rw = float(st.get("rw_assign_rate", 0.0))
            rw_hol = float(st.get("rw_assign_rate_holiday", 0.0))
            cnt = float(st.get("count_assigned", 0.0))
            last_days = float(st.get("last_assigned_days", 9999.0))
            cnt_norm = min(cnt / C, 1.0)
            recency_bonus = max(0.0, 1.0 - min(last_days, L) / L)
            holiday_bonus = holiday_weight * rw_hol if is_holiday else 0.0
            return float(p) + beta1 * rw + beta2 * cnt_norm + beta3 * recency_bonus + holiday_bonus

        explanation["decisionPath"].append("greedy_pick_with_soft_fairness_and_employee_opt_out_cap")

        try:
            hrs = (datetime.combine(to_date(s["date"]), parse_time_simple(s["end"])) -
                   datetime.combine(to_date(s["date"]), parse_time_simple(s["start"]))).seconds / 3600.0
        except Exception:
            hrs = 8.0
        wk = week_key_from_date_str(s["date"])

        scored: list[tuple[str, float, float, float, float]] = []
        fairness_skips_hardcap = []

        for uid, p in ranked:
            urec = users_by_id.get(uid, {})
            tp = (urec.get("timed_properties") or [{}])[0]
            soft_cap_uid = tp.get("weekly_hours", fairness_weekly_soft_cap_hours or 40.0)
            hard_cap_uid = tp.get("max_weekly_hours", None)
            if hard_cap_uid is None and fairness_opt_out_hard_cap_delta_hours is not None:
                hard_cap_uid = soft_cap_uid + fairness_opt_out_hard_cap_delta_hours

            current_week_hours = assigned_hours_by_user_week[(uid, wk)]
            new_hours = current_week_hours + hrs

            if hard_cap_uid is not None and new_hours > hard_cap_uid:
                fairness_skips_hardcap.append({
                    "userId": uid, "reason": "opt_out_hard_cap", "week": wk,
                    "current_week_hours": float(current_week_hours),
                    "hard_cap_uid": float(hard_cap_uid)
                })
                continue

            base = blended_score(uid, p)
            pen = fairness_penalty(new_hours, soft_cap_uid, hard_cap_uid)
            final = base - pen
            explanation["candidatesAfterFinalScoreOnly"].append({
                "userId": uid,
                "final_score": float(final)
            })
            scored.append((uid, final, base, pen, float(new_hours)))

        explanation["decisionPath"].append("scoring_unconstrained")
        unconstrained_ranked = score_candidates_for_shift(
            s, unconstrained_candidates, users_by_id, stats_by_user_ctx, model, iso_calibrator, features
        )
        stats_for_uid_uc = {uid: (stats_by_user_ctx.get(uid, {}) or {}).get(ctx, {}) for uid, _ in unconstrained_ranked}
        def blended_score_uc(uid: str, p: float) -> float:
            st = stats_for_uid_uc.get(uid, {}) or {}
            rw = float(st.get("rw_assign_rate", 0.0))
            rw_hol = float(st.get("rw_assign_rate_holiday", 0.0))
            cnt = float(st.get("count_assigned", 0.0))
            last_days = float(st.get("last_assigned_days", 9999.0))
            cnt_norm = min(cnt / C, 1.0)
            recency_bonus = max(0.0, 1.0 - min(last_days, L) / L)
            holiday_bonus = holiday_weight * rw_hol if is_holiday else 0.0
            return float(p) + beta1 * rw + beta2 * cnt_norm + beta3 * recency_bonus + holiday_bonus
        for uid, p in unconstrained_ranked:
            urec = users_by_id.get(uid, {})
            tp = (urec.get("timed_properties") or [{}])[0]
            soft_cap_uid = tp.get("weekly_hours", fairness_weekly_soft_cap_hours or 40.0)
            hard_cap_uid = tp.get("max_weekly_hours", None)
            if hard_cap_uid is None and fairness_opt_out_hard_cap_delta_hours is not None:
                hard_cap_uid = soft_cap_uid + fairness_opt_out_hard_cap_delta_hours
            current_week_hours = assigned_hours_by_user_week[(uid, wk)]
            new_hours = current_week_hours + hrs
            base_uc = blended_score_uc(uid, p)
            pen_uc = fairness_penalty(new_hours, soft_cap_uid, hard_cap_uid)
            final_uc = base_uc - pen_uc
            explanation["unconstrainedCandidatesAfterFinalScoreOnly"].append({
                "userId": uid,
                "final_score": float(final_uc)
            })

        if not scored:
            explanation["notes"].append("no candidate within employee opt-out hard cap (constrained)")
            if fairness_skips_hardcap:
                explanation["notes"].append({"hard_cap_skipped": fairness_skips_hardcap})
            explanation["unconstrainedCandidatesAfterFinalScoreOnly"].sort(key=lambda d: d["final_score"], reverse=True)
            report["shifts"].append(explanation)
            continue

        scored_sorted = sorted(scored, key=lambda t: t[1], reverse=True)
        chosen, final_score, base_score, penalty_applied, final_week_hours = scored_sorted[0]

        result[sid] = chosen
        assigned_by_user[chosen].append(sid)
        assigned_hours_by_user_week[(chosen, wk)] += hrs
        assigned_hours_by_user_global[chosen] += hrs
        all_assignments_by_user[chosen].append(s)

        explanation["chosen"] = {
            "userId": chosen,
            "score_after_penalty": float(final_score),
            "base_score": float(base_score),
            "fairness_penalty": float(penalty_applied),
            "week_key": wk,
            "week_hours_after": float(final_week_hours)
        }
        if fairness_skips_hardcap:
            explanation["notes"].append({"hard_cap_skipped": fairness_skips_hardcap})

        explanation["candidatesAfterFinalScoreOnly"].sort(key=lambda d: d["final_score"], reverse=True)
        explanation["unconstrainedCandidatesAfterFinalScoreOnly"].sort(key=lambda d: d["final_score"], reverse=True)
        report["shifts"].append(explanation)

    for uid in users_by_id.keys():
        report["usersSummary"][uid] = {
            "totalHours": float(assigned_hours_by_user_global.get(uid, 0.0)),
            "assignedCount": len(assigned_by_user.get(uid, []))
        }
    return result, report

# =========================
# Build output
# =========================

def build_assigned_output(report: dict) -> dict:
    shifts = []
    for s in report.get("shifts", []):
        cons = s.get("candidatesAfterFinalScoreOnly") or []
        cons = [{"userId": c["userId"], "final_score": float(c.get("final_score", 0.0))} for c in cons]
        cons.sort(key=lambda d: d["final_score"], reverse=True)

        unc = s.get("unconstrainedCandidatesAfterFinalScoreOnly") or []
        unc = [{"userId": c["userId"], "final_score": float(c.get("final_score", 0.0))} for c in unc]
        unc.sort(key=lambda d: d["final_score"], reverse=True)

        shifts.append({
            "shiftId": int(s.get("shiftId")),
            "constrainedCandidates": cons,
            "unconstrainedCandidates": unc
        })
    return {"shifts": shifts}

def assign_top_candidates_to_shifts(base_json_path: str,
                                    model_output_path: str,
                                    out_json_path: str,
                                    source_label: str = "ML_MODEL"):

    with open(base_json_path, encoding="utf-8") as f:
        base = json.load(f)
    with open(model_output_path, encoding="utf-8") as f:
        model = json.load(f)

    sp = base.get("shift_plan", {}) or {}
    base_assignments = sp.get("shift_assignments", []) or []

    existing_by_shift = {}
    for a in base_assignments:
        try:
            sid = int(a.get("shift_id"))
            existing_by_shift[sid] = {
                "shift_id": sid,
                "employee_uuid": a.get("employee_uuid"),
                "source": a.get("source", "UNKNOWN"),
            }
        except (TypeError, ValueError):
            continue

    top_by_shift = {}
    for entry in model.get("shifts", []) or []:
        sid = entry.get("shiftId")
        if sid is None:
            continue
        try:
            sid_int = int(sid)
        except (TypeError, ValueError):
            continue

        candidates = entry.get("constrainedCandidates") or entry.get("candidates") or []
        if not candidates:
            continue

        best = max(candidates, key=lambda c: c.get("final_score", float("-inf")))
        user_id = best.get("userId")
        if not user_id:
            continue
        top_by_shift[sid_int] = {
            "shift_id": sid_int,
            "employee_uuid": user_id,
            "source": source_label
        }

    known_shift_ids = set()
    for sh in sp.get("shifts", []) or []:
        try:
            known_shift_ids.add(int(sh.get("id")))
        except (TypeError, ValueError):
            pass

    filtered_top_by_shift = {sid: a for sid, a in top_by_shift.items() if sid in known_shift_ids}

    merged_assignments_by_shift = dict(existing_by_shift)
    for sid, a in filtered_top_by_shift.items():
        if sid not in merged_assignments_by_shift:
            merged_assignments_by_shift[sid] = a

    new_assignments = sorted(merged_assignments_by_shift.values(), key=lambda x: x["shift_id"])

    out_base = dict(base)
    out_sp = dict(sp)
    out_sp["shift_assignments"] = new_assignments
    out_base["shift_plan"] = out_sp

    os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(out_base, f, indent=2, ensure_ascii=False)

    return out_json_path
