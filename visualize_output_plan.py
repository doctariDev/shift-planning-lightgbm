#!/usr/bin/env python3
import json
import os
import shutil
from collections import defaultdict
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve

INPUT_JSON = "output/output_job.json"

CALENDER_OUT_DIR = "output/calender_renders"
DPI = 300

EMP_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf", "#393b79", "#637939",
    "#8c6d31", "#843c39", "#7b4173", "#a55194",
    "#6b6ecf", "#9c9ede", "#cedb9c", "#e7cb94"
]
UNASSIGNED_COLOR = "#B0B0B0"
HOLIDAY_BORDER_COLOR = "#800080"
GRID_COLOR = "#DDDDDD"

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def empty_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        return
    for name in os.listdir(path):
        fp = os.path.join(path, name)
        try:
            if os.path.isfile(fp) or os.path.islink(fp):
                os.remove(fp)
            else:
                shutil.rmtree(fp)
        except Exception as e:
            print(f"Warning: could not remove {fp}: {e}")

def save_fig(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=DPI)
    plt.close()


def plot_feature_importance(model, out_path: str, max_num_features=20):
    try:
        gains = model.feature_importance(importance_type="gain")
        names = model.feature_name()
        if gains is None or len(gains) == 0 or np.all(np.array(gains) == 0):
            plt.figure(figsize=(6,2))
            plt.text(0.5, 0.5, "No feature importance available", ha="center", va="center")
            plt.axis("off")
            save_fig(out_path)
            return
        imp = sorted(zip(names, gains), key=lambda x: x[1], reverse=True)[:max_num_features]
        labels = [i[0] for i in imp]
        values = [i[1] for i in imp]
        plt.figure(figsize=(8, max(3, 0.3*len(labels)+1)))
        plt.barh(labels[::-1], values[::-1])
        plt.xlabel("Gain")
        plt.title("Feature Importance (gain)")
        save_fig(out_path)
    except Exception as e:
        plt.figure(figsize=(6,2))
        plt.text(0.5, 0.5, f"Feature importance unavailable: {e}", ha="center", va="center", wrap=True)
        plt.axis("off")
        save_fig(out_path)

def plot_calibration(y_val, p_raw, p_cal, out_path: str):
    from sklearn.calibration import calibration_curve
    plt.figure(figsize=(6,5))
    for name, p in [("Raw", p_raw), ("Calibrated", p_cal)]:
        frac_pos, mean_pred = calibration_curve(y_val, p, n_bins=10, strategy="quantile")
        plt.plot(mean_pred, frac_pos, marker='o', label=name)
    plt.plot([0,1],[0,1], 'k--', alpha=0.5)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration curve")
    plt.legend()
    save_fig(out_path)

def plot_pr_roc(y_val, p_cal, out_pr_path: str, out_roc_path: str):
    prec, rec, _ = precision_recall_curve(y_val, p_cal)
    ap = average_precision_score(y_val, p_cal)
    plt.figure(figsize=(6,5))
    plt.plot(rec, prec)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"PR curve (AP={ap:.3f})")
    save_fig(out_pr_path)
    fpr, tpr, _ = roc_curve(y_val, p_cal)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr); plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC curve (AUC={roc_auc:.3f})")
    save_fig(out_roc_path)

def plot_hours_per_user(assigned_hours_by_user: dict, out_path: str):
    if not assigned_hours_by_user:
        return
    users = list(assigned_hours_by_user.keys())
    hours = [assigned_hours_by_user[u] for u in users]
    plt.figure(figsize=(8,4))
    plt.bar(users, hours)
    plt.ylabel("Assigned hours"); plt.title("Workload distribution")
    plt.xticks(rotation=45, ha='right')
    save_fig(out_path)

def plot_topk_for_shift(shift_id: str, top_list: list, out_path: str, k: int = 5):
    if not top_list:
        return
    data = top_list[:k]
    users = [d["userId"] for d in data][::-1]
    probs = [d["p_calibrated"] for d in data][::-1]
    plt.figure(figsize=(6,4))
    plt.barh(users, probs)
    plt.xlabel("Calibrated probability")
    plt.title(f"Top-{k} candidates for {shift_id}")
    save_fig(out_path)

def write_html_report(out_dir: str, run_title: str, summary: dict, images: dict, assignment_report_path: str):
    out_path = os.path.join(out_dir, "report.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"<html><body><h1>{run_title}</h1></body></html>")
    return out_path

def to_dt(s: str) -> datetime:
    return pd.to_datetime(s).to_pydatetime()

def fmt_hm(dt: datetime) -> str:
    return dt.strftime("%H:%M")

def daterange(start_date: date, end_date: date):
    cur = start_date
    while cur <= end_date:
        yield cur
        cur += timedelta(days=1)

def build_employee_maps(employees):
    id2name = {}
    for e in employees:
        full = f"{e.get('first_name','').strip()} {e.get('last_name','').strip()}".strip()
        id2name[e["uuid"]] = full if full else e["uuid"]
    emp_ids = sorted(id2name.keys())
    id2color = {uid: EMP_COLORS[i % len(EMP_COLORS)] for i, uid in enumerate(emp_ids)}
    id2color[None] = UNASSIGNED_COLOR
    return id2name, id2color

def load_output(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def build_workplace_name_map(shift_plan: dict, past_shift_plans: list):
    id2wp = {}
    for wp in (shift_plan.get("workplaces") or []):
        id2wp[int(wp["id"])] = wp.get("name") or f"Workplace {wp['id']}"
    for plan in past_shift_plans or []:
        for wp in (plan.get("workplaces") or []):
            id2wp[int(wp["id"])] = wp.get("name") or f"Workplace {wp['id']}"
    return id2wp

def build_shift_records(shifts, assignments, id2name, id2wpname):
    asg_by_shift = {int(a["shift_id"]): a for a in (assignments or [])}
    rows = []
    for sh in shifts or []:
        sid = int(sh["id"])
        start = to_dt(sh["start_date_time"])
        end = to_dt(sh["end_date_time"])
        wp_id = int(sh.get("workplace_id")) if sh.get("workplace_id") is not None else None
        asg = asg_by_shift.get(sid, {})
        emp = asg.get("employee_uuid")
        rows.append({
            "shift_id": sid,
            "workplace_id": wp_id,
            "workplace_name": id2wpname.get(wp_id, f"Workplace {wp_id}") if wp_id is not None else "Unknown",
            "shiftType": str(sh.get("shift_card_id") or "GEN"),
            "start": start,
            "end": end,
            "date": start.date(),
            "weekday": start.weekday(),
            "assigned": emp is not None,
            "employee_uuid": emp,
            "employee_name": id2name.get(emp, "Unassigned")
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        def last_name(full: str) -> str:
            if not full or full == "Unassigned":
                return "Unassigned"
            parts = full.strip().split()
            return parts[-1] if parts else full
        df["employee_last_name"] = df["employee_name"].apply(last_name)
        df = df.sort_values(["date", "workplace_name", "start"]).reset_index(drop=True)
    return df

def parse_holiday_days(holiday_list):
    days = set()
    for h in holiday_list or []:
        dstr = h.get("date") or h.get("start")
        if not dstr:
            continue
        try:
            days.add(pd.to_datetime(dstr).date())
        except Exception:
            continue
    return days

def compute_row_layout(df: pd.DataFrame, start_date: date, end_date: date,
                       cell_height_px: int = 110,
                       row_padding_top_px: int = 36,
                       row_padding_bottom_px: int = 36,
                       min_block_h_px: int = 40,
                       max_days: int = 31):
    all_days = list(daterange(start_date, end_date))
    if len(all_days) > max_days:
        all_days = all_days[:max_days]
    day_index = {d: i for i, d in enumerate(all_days)}
    cols = len(all_days)

    workplaces = sorted(df["workplace_name"].dropna().unique().tolist())

    grouped = defaultdict(list)
    for _, row in df.iterrows():
        d = row["date"]
        w = row["workplace_name"]
        if d in day_index and w in workplaces:
            grouped[(d, w)].append(row)

    row_heights_px = {}
    for w in workplaces:
        max_stack = 1
        for d in all_days:
            n = len(grouped.get((d, w), []))
            max_stack = max(max_stack, n)
        needed_blocks_h = max_stack * max(cell_height_px, min_block_h_px)
        row_heights_px[w] = needed_blocks_h + row_padding_top_px + row_padding_bottom_px

    return {
        "workplaces": workplaces,
        "day_index": day_index,
        "cols": cols,
        "grouped": grouped,
        "row_heights_px": row_heights_px,
        "all_days": all_days,
        "row_padding_top_px": row_padding_top_px,
        "row_padding_bottom_px": row_padding_bottom_px,
        "cell_height_px": cell_height_px,
        "min_block_h_px": min_block_h_px
    }


def draw_timeline_calendar(ax, start_date: date, end_date: date, df: pd.DataFrame,
                           id2color: dict, title: str, holidays_set=None, max_days: int = 31,
                           font_day: int = 12, font_label: int = 11,
                           cell_height_px: int = 110,
                           row_padding_top_px: int = 36,
                           row_padding_bottom_px: int = 36,
                           min_block_h_px: int = 40,
                           dpi: int = DPI):
    holidays_set = holidays_set or set()

    layout = compute_row_layout(df, start_date, end_date,
                                cell_height_px=cell_height_px,
                                row_padding_top_px=row_padding_top_px,
                                row_padding_bottom_px=row_padding_bottom_px,
                                min_block_h_px=min_block_h_px,
                                max_days=max_days)
    workplaces = layout["workplaces"]
    day_index = layout["day_index"]
    cols = layout["cols"]
    grouped = layout["grouped"]
    row_heights_px = layout["row_heights_px"]

    if not workplaces or cols == 0:
        ax.axis("off")
        ax.text(0.5, 0.5, "No shifts", ha="center", va="center")
        return

    total_height_px = sum(row_heights_px[w] for w in workplaces)

    y_bottom = {}
    cur_y = 0
    for w in reversed(workplaces):
        y_bottom[w] = cur_y
        cur_y += row_heights_px[w]

    ax.set_xlim(0, cols)
    ax.set_ylim(0, total_height_px)
    ax.axis("off")

    ax.text(0, total_height_px + 0.06 * total_height_px, title, fontsize=14, weight="bold",
            ha="left", va="bottom", transform=ax.transData)
    for d, x in day_index.items():
        ax.text(x + 0.5, total_height_px + 0.03 * total_height_px, f"{d.day}",
                fontsize=font_day, ha="center", va="bottom", transform=ax.transData)
        if d in holidays_set:
            ax.plot([x, x+1], [total_height_px, total_height_px], color=HOLIDAY_BORDER_COLOR, linewidth=3)

    for c in range(cols + 1):
        ax.plot([c, c], [0, total_height_px], color=GRID_COLOR, linewidth=0.8)

    for w in workplaces:
        y0 = y_bottom[w]
        h_px = row_heights_px[w]
        ax.plot([0, cols], [y0, y0], color=GRID_COLOR, linewidth=0.8)
        ax.text(-0.1, y0 + h_px/2, w, fontsize=font_day, ha="right", va="center", transform=ax.transData)

    pad_x = 0.10

    for (d, w), items in grouped.items():
        x = day_index[d]
        y0 = y_bottom[w]
        h_px = row_heights_px[w]
        usable_h_px = max(1, h_px - (row_padding_top_px + row_padding_bottom_px))
        n = max(1, len(items))

        nominal = cell_height_px
        block_h_px = max(min_block_h_px, min(nominal, usable_h_px / n))

        y_cursor = y0 + h_px - row_padding_top_px - block_h_px

        items = sorted(items, key=lambda r: r["start"])
        for it in items:
            color = id2color.get(it["employee_uuid"], UNASSIGNED_COLOR)

            rect_x = x + pad_x
            rect_w = 1 - 2 * pad_x
            rect_y = max(y0 + row_padding_bottom_px, y_cursor)
            rect_h = block_h_px

            top_limit = y0 + h_px - row_padding_top_px
            if rect_y + rect_h > top_limit:
                rect_h = max(min_block_h_px, top_limit - rect_y)

            ax.add_patch(plt.Rectangle((rect_x, rect_y),
                                       rect_w, rect_h,
                                       color=color, alpha=0.95, ec="black", lw=0.3))

            time_part = f"{fmt_hm(it['start'])}-{fmt_hm(it['end'])}"
            name_part = it["employee_last_name"] if it["assigned"] else "Unassigned"
            fsize = font_label if n <= 2 else max(9, font_label - 1)

            text_x = rect_x + 0.02
            text_y_center = rect_y + rect_h / 2.0
            line_spacing_px = max(6, rect_h * 0.28)

            ax.text(text_x, text_y_center + line_spacing_px/2,
                    time_part, fontsize=fsize, color="white",
                    ha="left", va="center", transform=ax.transData)

            ax.text(text_x, text_y_center - line_spacing_px/2,
                    name_part, fontsize=fsize, color="white",
                    ha="left", va="center", transform=ax.transData)

            y_cursor -= (rect_h + 10)

def render_timeline_for_plan(plan_name: str, df: pd.DataFrame, planning_period: dict,
                             id2color: dict, out_dir: str, holidays_set=None, max_days: int = 31,
                             per_day_inch: float = 2.0,
                             cell_height_px: int = 110,
                             row_padding_top_px: int = 36,
                             row_padding_bottom_px: int = 36,
                             min_block_h_px: int = 40):
    if df.empty:
        return []
    holidays_set = holidays_set or set()

    start = pd.to_datetime(planning_period.get("start_date")).date() if planning_period else df["date"].min()
    end = pd.to_datetime(planning_period.get("end_date")).date() if planning_period else df["date"].max()

    outputs = []
    cur = start
    while cur <= end:
        page_end = min(end, cur + timedelta(days=max_days - 1))
        all_days = list(daterange(cur, page_end))
        if len(all_days) > max_days:
            all_days = all_days[:max_days]
        days_on_page = len(all_days)

        fig_w = max(18, per_day_inch * days_on_page)

        layout = compute_row_layout(df, cur, page_end,
                                    cell_height_px=cell_height_px,
                                    row_padding_top_px=row_padding_top_px,
                                    row_padding_bottom_px=row_padding_bottom_px,
                                    min_block_h_px=min_block_h_px,
                                    max_days=max_days)
        total_height_px = sum(layout["row_heights_px"][w] for w in layout["workplaces"])
        fig_h = max(6, total_height_px / DPI + 1.8)
        fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
        title = f"{plan_name} – {cur.strftime('%Y-%m-%d')} to {page_end.strftime('%Y-%m-%d')}"
        draw_timeline_calendar(ax, cur, page_end, df, id2color, title,
                               holidays_set=holidays_set, max_days=max_days,
                               font_day=13, font_label=12,
                               cell_height_px=cell_height_px,
                               row_padding_top_px=row_padding_top_px,
                               row_padding_bottom_px=row_padding_bottom_px,
                               min_block_h_px=min_block_h_px,
                               dpi=DPI)
        fname = f"{plan_name.replace(' ', '_').lower()}_{cur.strftime('%Y%m%d')}_{page_end.strftime('%Y%m%d')}.png"
        out_path = os.path.join(out_dir, fname)
        plt.tight_layout()
        plt.savefig(out_path, dpi=DPI)
        plt.close(fig)
        outputs.append(out_path)
        cur = page_end + timedelta(days=1)

    return outputs

def visualize_plans(json_path: str = None):
    if json_path is None:
        json_path = INPUT_JSON
    if not isinstance(json_path, (str, os.PathLike)):
        raise TypeError("visualize_plans expects a file path string; got a different type.")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file does not exist: {json_path}")

    ensure_dir(CALENDER_OUT_DIR)
    empty_dir(CALENDER_OUT_DIR)

    data = load_output(json_path)

    sp = data.get("shift_plan", {}) or {}
    past = data.get("past_shift_plans", []) or []
    employees = sp.get("employees", []) or []
    id2name, id2color = build_employee_maps(employees)
    id2wpname = build_workplace_name_map(sp, past)

    target_df = build_shift_records(
        sp.get("shifts", []) or [],
        sp.get("shift_assignments", []) or [],
        id2name, id2wpname
    )
    target_holidays = parse_holiday_days(sp.get("public_holidays") or [])
    target_period = sp.get("planning_period") or {}

    if target_df.empty:
        print("Warning: Target plan has no shifts to render.")

    generated_files = []
    for i, plan in enumerate(past):
        plan_label = f"Past Plan {plan.get('id', i+1)}"
        hist_df = build_shift_records(
            plan.get("shifts", []) or [],
            plan.get("shift_assignments", []) or [],
            id2name, id2wpname
        )
        plan_holidays = parse_holiday_days(plan.get("public_holidays") or [])
        plan_period = plan.get("planning_period") or {}
        if hist_df.empty:
            print(f"Info: {plan_label} has no shifts.")
        else:
            generated_files += render_timeline_for_plan(
                plan_label, hist_df, plan_period, id2color,
                CALENDER_OUT_DIR, holidays_set=plan_holidays, max_days=31,
                per_day_inch=2.2,
                cell_height_px=110,
                row_padding_top_px=36,
                row_padding_bottom_px=36,
                min_block_h_px=48
            )

    generated_files += render_timeline_for_plan(
        "Target Plan", target_df, target_period, id2color,
        CALENDER_OUT_DIR, holidays_set=target_holidays, max_days=31,
        per_day_inch=2.2,
        cell_height_px=110,
        row_padding_top_px=36,
        row_padding_bottom_px=36,
        min_block_h_px=48
    )

    if not generated_files:
        print("No timeline images were generated (no shifts in any plan).")
    else:
        print("Generated images:")
        for p in generated_files:
            print(f" - {p}")

if __name__ == "__main__":
    json_path = "output/prc_with_assignments.json"
    visualize_plans(json_path)
