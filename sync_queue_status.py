#!/usr/bin/env python3
"""
Live Queue Status Synchronizer
Continuously updates results/overnight_queue_status.txt with live progress,
samples processed, remaining samples, ETA, and exact estimated finish clock times
for the remaining tasks in the overnight queue.
"""

import os
import sys
import time
import re
import json
from datetime import datetime, timedelta

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
STATUS_FILE = os.path.join(RESULTS_DIR, "overnight_queue_status.txt")

QUEUE_TASKS = [
    {
        "id": "Task 8",
        "name": "Clean Model + Clean + Equi Def",
        "target": "37.95%",
        "prefix": "baseline_drn_d_22_clean_equidefense",
        "nominal_duration_min": 70.0,
    },
    {
        "id": "Task 10",
        "name": "Clean Model + PGD-50 + Equi Def",
        "target": "30.76%",
        "prefix": "baseline_drn_d_22_pgd_equidefense",
        "nominal_duration_min": 82.0,
    },
    {
        "id": "Task 6",
        "name": "Adv Model + BPDA + Equi Def",
        "target": "29.83%",
        "prefix": "baseline_advtrain_drn_d_22_bpda_equidefense",
        "nominal_duration_min": 780.0,
    }
]


def parse_progress_file(filepath):
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        m_samples = re.search(r"Samples:\s*(\d+)/(\d+)\s*\(([\d\.]+)%\)", content)
        m_batch = re.search(r"Batch:\s*(\d+)/(\d+)", content)
        m_eta = re.search(r"ETA:\s*([\d\.]+)\s*min", content)
        m_elapsed = re.search(r"Elapsed:\s*([\d\.]+)\s*min", content)
        m_latency = re.search(r"Latency:\s*([\d\.]+)\s*s", content)
        m_speed = re.search(r"Speed:\s*([\d\.]+)\s*FPS", content)
        m_clean = re.search(r"Clean.*:\s*([\d\.]+)%", content)
        m_def = re.search(r"(?:BPDA \+ Equi Def|Clean \+ Equi Def|Equi Def|Defended|Def):\s*([\d\.]+)%", content, re.IGNORECASE)

        if not m_samples:
            return None

        samples = int(m_samples.group(1))
        total = int(m_samples.group(2))
        percent = float(m_samples.group(3))
        eta_min = float(m_eta.group(1)) if m_eta else None
        elapsed_min = float(m_elapsed.group(1)) if m_elapsed else None

        return {
            "samples": samples,
            "total": total,
            "percent": percent,
            "remaining_samples": total - samples,
            "eta_min": eta_min,
            "elapsed_min": elapsed_min,
            "latency": m_latency.group(1) if m_latency else "--",
            "speed": m_speed.group(1) if m_speed else "--",
            "clean_miou": m_clean.group(1) if m_clean else "--",
            "def_miou": m_def.group(1) if m_def else "--",
            "batch": f"{m_batch.group(1)}/{m_batch.group(2)}" if m_batch else "--"
        }
    except Exception:
        return None


def is_task_completed(prefix):
    metrics_path = os.path.join(RESULTS_DIR, f"{prefix}_metrics.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("samples_evaluated") == 500 or data.get("status") == "completed":
                return True
        except Exception:
            pass
    summary_path = os.path.join(RESULTS_DIR, f"{prefix}_summary.txt")
    if not os.path.exists(summary_path):
        return False
    try:
        with open(summary_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        return bool(re.search(r"Samples Evaluated:\s*500", text, re.IGNORECASE) or "500 samples" in text.lower())
    except Exception:
        return False


def format_time_est(dt):
    today = datetime.now().date()
    if dt.date() == today:
        return dt.strftime("Today at %I:%M %p")
    elif dt.date() == today + timedelta(days=1):
        return dt.strftime("Tomorrow at %I:%M %p")
    else:
        return dt.strftime("%b %d at %I:%M %p")


def format_duration(minutes):
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = int(minutes // 60)
    mins = int(round(minutes % 60))
    return f"{hours}h {mins:02d}m"


def render_progress_bar(percent, length=28):
    filled = int(round(length * percent / 100.0))
    bar = "=" * max(0, filled - 1) + (">" if 0 < filled < length else "=" if filled == length else "")
    bar = bar.ljust(length, "-")
    return f"[{bar}]"


def generate_status_text():
    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d %I:%M:%S %p")

    completed_count = 0
    running_task_idx = None
    task_states = []

    for idx, t in enumerate(QUEUE_TASKS):
        prefix = t["prefix"]
        if is_task_completed(prefix):
            completed_count += 1
            task_states.append({
                "task": t,
                "status": "COMPLETED",
                "samples": 500,
                "total": 500,
                "percent": 100.0,
                "rem_samples": 0,
                "rem_duration_min": 0.0,
                "prog_info": None
            })
        else:
            prog_info = parse_progress_file(os.path.join(RESULTS_DIR, f"{prefix}_progress.txt"))
            if prog_info and prog_info["samples"] is not None and running_task_idx is None:
                running_task_idx = idx
                rem_min = prog_info["eta_min"] if prog_info["eta_min"] is not None else t["nominal_duration_min"]
                task_states.append({
                    "task": t,
                    "status": "RUNNING",
                    "samples": prog_info["samples"],
                    "total": prog_info["total"],
                    "percent": prog_info["percent"],
                    "rem_samples": prog_info["remaining_samples"],
                    "rem_duration_min": rem_min,
                    "prog_info": prog_info
                })
            else:
                task_states.append({
                    "task": t,
                    "status": "QUEUED",
                    "samples": 0,
                    "total": 500,
                    "percent": 0.0,
                    "rem_samples": 500,
                    "rem_duration_min": t["nominal_duration_min"],
                    "prog_info": None
                })

    # Timeline calculation
    current_offset = 0.0
    for state in task_states:
        if state["status"] == "COMPLETED":
            state["finish_est"] = "COMPLETED"
            state["time_left_str"] = "Done (0m left)"
        elif state["status"] == "RUNNING":
            current_offset += state["rem_duration_min"]
            finish_dt = now + timedelta(minutes=current_offset)
            state["finish_est"] = format_time_est(finish_dt)
            state["time_left_str"] = f"{format_duration(state['rem_duration_min'])} left"
        elif state["status"] == "QUEUED":
            current_offset += state["rem_duration_min"]
            finish_dt = now + timedelta(minutes=current_offset)
            state["finish_est"] = format_time_est(finish_dt)
            state["time_left_str"] = f"~{format_duration(state['rem_duration_min'])} left"

    total_queue_time_left = current_offset
    overall_finish_dt = now + timedelta(minutes=total_queue_time_left)

    active_state = task_states[running_task_idx] if running_task_idx is not None else None

    # Generate document
    lines = []
    lines.append("=" * 96)
    lines.append("        OVERNIGHT REPRODUCTION QUEUE STATUS & TIME ESTIMATE TRACKER")
    lines.append("=" * 96)
    lines.append("")
    lines.append(f"  Current Time:        {now_str} (Auto-refreshes every 10s)")
    lines.append(f"  Queue Progress:      {completed_count} / {len(QUEUE_TASKS)} tasks completed (Shortest-Job-First)")
    lines.append(f"  Total Queue Left:    ~{format_duration(total_queue_time_left)}")
    lines.append(f"  Est. Final Finish:   {format_time_est(overall_finish_dt)}")
    lines.append("")
    lines.append("=" * 96)
    lines.append("                       REMAINING TASKS FINISH TIMELINE")
    lines.append("=" * 96)
    header = f"{'Task':<9}{'Status':<11}{'Completed':<18}{'Remaining':<18}{'Time Left':<18}{'Est. Finish Time':<22}"
    lines.append(header)
    lines.append("-" * 96)

    for s in task_states:
        t = s["task"]
        prog_str = f"{s['samples']}/{s['total']} ({s['percent']:.1f}%)" if s['status'] != 'COMPLETED' else "500/500 (100%)"
        rem_str = f"{s['rem_samples']} images left" if s['status'] != 'COMPLETED' else "0 images left"
        lines.append(f"{t['id']:<9}{s['status']:<11}{prog_str:<18}{rem_str:<18}{s['time_left_str']:<18}{s['finish_est']:<22}")

    lines.append("-" * 96)
    lines.append("")

    if active_state and active_state["prog_info"]:
        info = active_state["prog_info"]
        lines.append("=" * 96)
        lines.append(f"               CURRENTLY ACTIVE: {active_state['task']['id']} ({active_state['task']['name']})")
        lines.append("=" * 96)
        p_bar = render_progress_bar(info['percent'], length=30)
        lines.append(f"  Progress Bar:        {p_bar} {info['samples']}/500 images ({info['percent']:.1f}%)")
        lines.append(f"  Images Left:         {info['remaining_samples']} images remaining")
        lines.append(f"  Time Elapsed:        {info['elapsed_min']:.1f} mins")
        lines.append(f"  Time Remaining:      {info['eta_min']:.1f} mins (~{format_duration(info['eta_min'])})")
        lines.append(f"  Est. Finish Time:    {active_state['finish_est']}")
        lines.append(f"  Batch Latency:       {info['latency']}s per batch of 2 images ({info['speed']} FPS)")
        lines.append(f"  Running Metrics:     Clean mIoU: {info['clean_miou']}% | Defended mIoU: {info['def_miou']}%")
        lines.append(f"  Paper Target:        {active_state['task']['target']}")
        lines.append(f"  GitHub Action:       Auto-commit and git push will trigger the instant this finishes!")
        lines.append("=" * 96)
        lines.append("")
    elif completed_count == len(QUEUE_TASKS):
        lines.append("=" * 96)
        lines.append("  ALL QUEUE TASKS COMPLETED & PUSHED TO GITHUB SUCCESSFULLY!")
        lines.append("=" * 96)
        lines.append("")

    return "\n".join(lines)


STATUS_FILES = [
    os.path.join(RESULTS_DIR, "overnight_queue_status.txt"),
    os.path.join(RESULTS_DIR, "LIVE_QUEUE_STATUS.txt"),
    os.path.join(RESULTS_DIR, "LIVE_QUEUE_STATUS.md"),
]


def update_status_file():
    content = generate_status_text()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for path in STATUS_FILES:
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
                f.flush()
        except Exception:
            pass


def main():
    while True:
        try:
            update_status_file()
        except Exception as e:
            pass
        time.sleep(5)


if __name__ == "__main__":
    main()
