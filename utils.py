import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, date as date_cls, time as time_cls
import numpy as np
import pandas as pd
from collections import defaultdict
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score
import warnings
import os
import joblib

warnings.filterwarnings("ignore")

# ======== GLOBAL FEATURE LISTS =========

DEFAULT_FEATURES = [
    "unit_tags", "workplace_id", "shiftType", "weekday", "hour", "duration",
    "isHoliday", "rw_assign_rate", "count_assigned", "last_assigned_days", "userFTE",
    "weeks_since_last_ctx_wd", "worked_last_1w_ctx_wd", "worked_last_2w_ctx_wd", "freq_ctx_wd_8w"
]
DEFAULT_CAT_FEATURES = ["unit_tags", "workplace_id", "shiftType"]

# =========================
# Util functions
# =========================

def dt_parse_iso(s: str, tz: Optional[str]) -> datetime:
    dt = pd.to_datetime(s)
    return dt.to_pydatetime()

def parse_time_simple(tstr: str) -> time_cls:
    parts = list(map(int, tstr.split(":")))
    if len(parts) == 2:
        return time_cls(parts[0], parts[1])
    return time_cls(parts[0], parts[1], parts[2])

def to_date(obj) -> Optional[date_cls]:
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
        return obj.date()
    if isinstance(obj, date_cls):
        return obj
    raise TypeError(f"Unsupported date type: {type(obj)}")

def iso_week_index_from_date(d: date_cls) -> int:
    y, w, _ = d.isocalendar()
    return int(y) * 100 + int(w)

def rolling_weeks_freq(weeks_sorted: list[int], current_week_idx: int, window: int = 8) -> float:
    if not weeks_sorted:
        return 0.0
    lo = current_week_idx - window
    cnt = sum(1 for wk in weeks_sorted if lo <= wk < current_week_idx)
    return cnt / float(window)

def user_qualified(user: Dict[str, Any], required_quals: List[int]) -> bool:
    uq = {q["id"] for q in user.get("qualifications", [])}
    return set(required_quals or []).issubset(uq)

def conflicts_with_parallel(shift: Dict[str, Any], user_id: str, assigned_by_user: Dict[str, List[int]], shift_index: Dict[int, Dict[str, Any]]) -> bool:
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

def parse_holiday_days(holiday_list: List[Dict[str, Any]]) -> set:
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
# Adapters
# =========================

def adapt_past_plans_to_frames(data: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
                "unit_tags": str(sh.get("unit_tags")),
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

def adapt_target_plan_to_frames(data: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]], Dict[str, Dict[str, Any]], Optional[str]]:
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
            "unit_tags": str(sh.get("unit_tags")),
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
    lam: float = 0.85) -> Dict[str, Dict[Tuple, Dict[str, float]]]:
    if hist_shifts_df.empty:
        return {}
    df = hist_shifts_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    now = df["date"].max()
    now_dt = pd.to_datetime(now) if pd.notnull(now) else pd.Timestamp(datetime.utcnow())


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

    df["context"] = df.apply(lambda r: (r["unit_tags"], r["shiftType"], r["weekday"]), axis=1)
    context_by_shift = dict(zip(df["id"], df["context"]))

    if assignments_df.empty:
        return stats

    merged = assignments_df.merge(df[["id", "unit_tags", "shiftType", "weekday", "date", "isHoliday"]],
                                left_on="shiftId", right_on="id", how="left")
    merged["date"] = pd.to_datetime(merged["date"])

    for _, row in merged.iterrows():
        uid = row["userId"]
        sid = row["shiftId"]
        if sid not in context_by_shift:
            continue
        ctx = context_by_shift[sid]
        date_dt = row["date"]
        if pd.isnull(date_dt):
            continue
        w = recency_weight(date_dt, now_dt, lam=lam, unit="week")
        sh_is_hol = int(row.get("isHoliday", 0)) == 1
        s = stats[uid][ctx]

        s["rw_num"] += w
        s["rw_denom"] += w
        s["count_assigned"] += 1
        s["count_occurrences"] += 1
        s["last_assigned_days"] = min(s["last_assigned_days"], (now_dt - date_dt).days)

        wk_idx = iso_week_index_from_date(date_dt.date())
        s["weeks_worked"].append(wk_idx)

        if sh_is_hol:
            s["rw_num_holiday"] += w
            s["rw_denom_holiday"] += w
            s["count_assigned_holiday"] += 1
            s["count_occurrences_holiday"] += 1

    for uid, ctxs in stats.items():
        for ctx, s in ctxs.items():
            s["rw_assign_rate"] = (s["rw_num"] / max(s["rw_denom"], 1e-6))
            if s["rw_denom_holiday"] > 0:
                s["rw_assign_rate_holiday"] = s["rw_num_holiday"] / max(s["rw_denom_holiday"], 1e-6)
            else:
                s["rw_assign_rate_holiday"] = 0.0
            s["weeks_worked"] = sorted(set(s["weeks_worked"]))

    return stats

def build_training_data(hist_shifts_df: pd.DataFrame,
    assignments_df: pd.DataFrame,
    users_df: pd.DataFrame,
    stats_by_user_ctx: Dict[str, Dict[Tuple, Dict[str, float]]],
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
        ctx = (sm.get("unit_tags"), sm.get("shiftType"), sm.get("weekday"))
        start_s = sm.get("start", "08:00")
        end_s = sm.get("end", "16:00")
        try:
            hour_val = int(str(start_s)[:2])
        except Exception:
            hour_val = 8
        try:
            s_date = to_date(sm.get("date"))
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
            pfeats = periodic_feats(uid, ctx, sm.get("date"))
            rows.append({
                "shiftId": sid, "userId": uid, "y": 1,
                "unit_tags": sm.get("unit_tags"),
                "workplace_id": sm.get("workplace_id"),  # <-- added here
                "shiftType": sm.get("shiftType"),
                "weekday": sm.get("weekday"),
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
                pfeats = periodic_feats(uid, ctx, sm.get("date"))
                rows.append({
                    "shiftId": sid, "userId": uid, "y": 0,
                    "unit_tags": sm.get("unit_tags"),
                    "workplace_id": sm.get("workplace_id"),  # <-- added here
                    "shiftType": sm.get("shiftType"),
                    "weekday": sm.get("weekday"),
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
    df["workplace_id"] = df["workplace_id"].astype("category")  # <-- Ensure this stays
    return df

# =========================
# Model training
# =========================

def train_lgb_full(
    df: pd.DataFrame,
    features: List[str] = DEFAULT_FEATURES,
    cat_features: List[str] = DEFAULT_CAT_FEATURES,
    params: dict = None,
    num_boost_round: int = 500,
    test_size: float = 0.25,
    random_state: int = 42
) -> Tuple[Any, Dict[str, List[str]], List[str], Tuple[pd.DataFrame, pd.Series]]:
    """Trains a LightGBM model on the given DataFrame."""
    X = df[features].copy()
    y = df["y"].copy()
    for c in cat_features:
        X[c] = X[c].astype("category")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    #category_levels = capture_category_levels(X, cat_features)
    lgb_train = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_features, free_raw_data=False)

    params = params or {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": 50,
        "verbosity": -1,
        "seed": random_state,
    }
    booster = lgb.train(params, lgb_train, num_boost_round=num_boost_round)
    return booster, features, (X_val, y_val)

def calibrate_isotonic(
    booster: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series
) -> Tuple[IsotonicRegression, dict]:
    p_raw_val = booster.predict(X_val)
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_raw_val, y_val)
    auc_val = roc_auc_score(y_val, p_raw_val)
    ap_val = average_precision_score(y_val, p_raw_val)
    brier = brier_score_loss(y_val, iso.predict(p_raw_val))
    metrics = {"auc": float(auc_val), "ap": float(ap_val), "brier": float(brier)}
    print(f"[Validation] AUC={auc_val:.3f} AP={ap_val:.3f} Brier={brier:.3f}")
    return iso, metrics

def continue_training(booster, df, num_additional_rounds=200, features=None, cat_features=None):
    features = features or DEFAULT_FEATURES
    cat_features = cat_features or DEFAULT_CAT_FEATURES
    X = df[features]
    y = df["y"]
    for c in cat_features:
        if X[c].dtype.name != "category":
            X[c] = X[c].astype("category")
    lgb_train = lgb.Dataset(X, label=y, categorical_feature=cat_features, free_raw_data=False)
    params = booster.params if hasattr(booster, "params") else {
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
    booster = lgb.train(
        params,
        lgb_train,
        num_boost_round=num_additional_rounds,
        init_model=booster
    )
    return booster

def recalibrate(booster, df, features=None, cat_features=None):
    features = features or DEFAULT_FEATURES
    cat_features = cat_features or DEFAULT_CAT_FEATURES
    X = df[features].copy()
    y = df["y"].copy()
    for c in cat_features:
        X[c] = X[c].astype("category")
    _, X_val, _, y_val = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    p_raw_val = booster.predict(X_val)
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_raw_val, y_val)
    auc_val = roc_auc_score(y_val, p_raw_val)
    ap_val = average_precision_score(y_val, p_raw_val)
    brier = brier_score_loss(y_val, iso.predict(p_raw_val))
    print(f"[Recalibration] AUC={auc_val:.3f} AP={ap_val:.3f} Brier={brier:.3f}")
    return iso


# =========================
# Scoring and assignment
# =========================

def score_candidates_for_shift(shift: Dict[str, Any],
                               candidate_ids: List[str],
                               users_by_id: Dict[str, Dict[str, Any]],
                               stats_by_user_ctx: Dict[str, Dict[Tuple, Dict[str, float]]],
                               model, iso_calibrator, features: List[str]) -> List[Tuple[str, float]]:
    ctx = (shift["unit_tags"], shift["workplace_id"], shift["shiftType"], shift["weekday"])
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
            "unit_tags": shift["unit_tags"], 
            "workplace_id": shift["workplace_id"],
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
    X["workplace_id"] = X["workplace_id"].astype("category")
    X["shiftType"] = X["shiftType"].astype("category")
    p_raw = model.predict(X[features])
    p = iso_calibrator.predict(p_raw)
    return list(zip(idx, p))

def calculate_assignment_scores(
    target_shifts: List[Dict[str, Any]],
    users_by_id: Dict[str, Dict[str, Any]],
    shift_index: Dict[int, Dict[str, Any]],
    model, iso_calibrator, features,
    stats_by_user_ctx: Dict[str, Dict[Tuple, Dict[str, float]]],
    fairness_weekly_soft_cap_hours: Optional[float] = None,
    fairness_opt_out_hard_cap_delta_hours: Optional[float] = None,
    customer_tz: Optional[str] = None,
    top_k: int = 5
) -> Dict[int, Dict[str, List[Dict[str, float]]]]:

    from collections import defaultdict
    from datetime import datetime

    assigned_hours_by_user_week: Dict[Tuple[str, str], float] = defaultdict(float)

    def week_key_from_date_str(dstr: str) -> str:
        d = to_date(dstr)
        y, w, _ = d.isocalendar()
        return f"{y}-W{w:02d}"

    def fairness_penalty(new_hours: float, soft_cap: Optional[float], hard_cap: Optional[float]) -> float:
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

    output: Dict[int, Dict[str, List[Dict[str, float]]]] = {}

    for s in target_shifts:
        sid = s["id"]
        output[sid] = {"constrained": [], "unconstrained": []}

        preplanned_user = s.get("preplannedUserId")
        all_user_ids = list(users_by_id.keys())
        unconstrained_candidates = [uid for uid in all_user_ids if uid != preplanned_user]

        ctx = (s["unit_tags"], s["shiftType"], s["weekday"])
        is_holiday = int(s.get("isHoliday", 0)) == 1

        try:
            hrs = (
                datetime.combine(to_date(s["date"]), parse_time_simple(s["end"])) -
                datetime.combine(to_date(s["date"]), parse_time_simple(s["start"]))
            ).seconds / 3600.0
        except Exception:
            hrs = 8.0

        wk = week_key_from_date_str(s["date"])

        # -------------------------
        # UNCONSTRAINED SCORING
        # -------------------------
        unconstrained_ranked = score_candidates_for_shift(
            s, unconstrained_candidates, users_by_id,
            stats_by_user_ctx, model, iso_calibrator, features
        )

        stats_uc = {
            uid: (stats_by_user_ctx.get(uid, {}) or {}).get(ctx, {})
            for uid, _ in unconstrained_ranked
        }

        beta1, beta2, beta3 = 0.10, 0.05, 0.05
        C, L = 5.0, 60.0
        holiday_weight = 0.08

        def blended_uc(uid: str, p: float) -> float:
            st = stats_uc.get(uid, {}) or {}
            cnt_norm = min(float(st.get("count_assigned", 0.0)) / C, 1.0)
            recency = max(0.0, 1.0 - min(float(st.get("last_assigned_days", 9999)), L) / L)
            holiday = holiday_weight * float(st.get("rw_assign_rate_holiday", 0.0)) if is_holiday else 0.0
            return float(p) + beta1 * float(st.get("rw_assign_rate", 0.0)) + beta2 * cnt_norm + beta3 * recency + holiday

        for uid, p in unconstrained_ranked:
            urec = users_by_id.get(uid, {})
            tp = (urec.get("timed_properties") or [{}])[0]
            soft = tp.get("weekly_hours", fairness_weekly_soft_cap_hours or 40.0)
            hard = tp.get("max_weekly_hours")
            if hard is None and fairness_opt_out_hard_cap_delta_hours is not None:
                hard = soft + fairness_opt_out_hard_cap_delta_hours

            new_hours = assigned_hours_by_user_week[(uid, wk)] + hrs
            final = blended_uc(uid, p) - fairness_penalty(new_hours, soft, hard)

            output[sid]["unconstrained"].append({
                "userId": uid,
                "final_score": float(final)
            })

        output[sid]["unconstrained"].sort(key=lambda d: d["final_score"], reverse=True)

        # -------------------------
        # CONSTRAINED SCORING
        # -------------------------
        ranked = score_candidates_for_shift(
            s, [], users_by_id,
            stats_by_user_ctx, model, iso_calibrator, features
        )

        stats_c = {
            uid: (stats_by_user_ctx.get(uid, {}) or {}).get(ctx, {})
            for uid, _ in ranked
        }

        def blended_c(uid: str, p: float) -> float:
            st = stats_c.get(uid, {}) or {}
            cnt_norm = min(float(st.get("count_assigned", 0.0)) / C, 1.0)
            recency = max(0.0, 1.0 - min(float(st.get("last_assigned_days", 9999)), L) / L)
            holiday = holiday_weight * float(st.get("rw_assign_rate_holiday", 0.0)) if is_holiday else 0.0
            return float(p) + beta1 * float(st.get("rw_assign_rate", 0.0)) + beta2 * cnt_norm + beta3 * recency + holiday

        for uid, p in ranked:
            urec = users_by_id.get(uid, {})
            tp = (urec.get("timed_properties") or [{}])[0]
            soft = tp.get("weekly_hours", fairness_weekly_soft_cap_hours or 40.0)
            hard = tp.get("max_weekly_hours")
            if hard is None and fairness_opt_out_hard_cap_delta_hours is not None:
                hard = soft + fairness_opt_out_hard_cap_delta_hours

            new_hours = assigned_hours_by_user_week[(uid, wk)] + hrs
            if hard is not None and new_hours > hard:
                continue

            final = blended_c(uid, p) - fairness_penalty(new_hours, soft, hard)

            output[sid]["constrained"].append({
                "userId": uid,
                "final_score": float(final)
            })

        output[sid]["constrained"].sort(key=lambda d: d["final_score"], reverse=True)

    return output

# =========== SAVING & LOADING MODELS ===========

def save_model_bundle(model, iso, features, save_dir="model_bundle"):
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "lgb_model.txt")
    model.save_model(model_path)
    iso_path = os.path.join(save_dir, "isotonic_calibrator.pkl")
    joblib.dump(iso, iso_path)
    meta = {
        "features": features,
        "lightgbm_params": model.params if hasattr(model, "params") else None
    }
    meta_path = os.path.join(save_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return {"model": model_path, "iso": iso_path, "meta": meta_path}

def load_model_bundle(save_dir="model_bundle"):
    model_path = os.path.join(save_dir, "lgb_model.txt")
    booster = lgb.Booster(model_file=model_path)
    iso_path = os.path.join(save_dir, "isotonic_calibrator.pkl")
    iso = joblib.load(iso_path)
    meta_path = os.path.join(save_dir, "metadata.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    features = meta["features"]
    return booster, iso, features

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

    with open(base_json_path, "r", encoding="utf-8") as f:
        base = json.load(f)
    with open(model_output_path, "r", encoding="utf-8") as f:
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

        candidates = entry.get("candidates") or []
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