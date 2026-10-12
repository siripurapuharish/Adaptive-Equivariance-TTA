#!/usr/bin/env python3
"""
Master Overnight Queue Runner (Shortest Job First)
Executes all remaining reproduction tasks sequentially, auto-commits each to Git, and pushes to GitHub.

Queue:
  1. Task 8:  Clean Model + Clean + Equivariance Defense       (~1 hr 10 mins)  [Target: 37.95%]
  2. Task 10: Clean Model + PGD-50 + Equivariance Defense      (~1 hr 22 mins)  [Target: 30.76%]
  3. Task 6:  Adv-Trained + BPDA + Equivariance Defense        (~13.0 hours)    [Target: 29.83%]

Total Estimated Queue Runtime: ~15.5 hours.
"""

import os
import sys
import time
import subprocess
import json
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
STATUS_FILE = os.path.join(RESULTS_DIR, "overnight_queue_status.txt")

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def update_status(queue_state: dict):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(STATUS_FILE, "w") as f:
        f.write("=" * 75 + "\n")
        f.write("   OVERNIGHT REPRODUCTION QUEUE STATUS (SHORTEST JOB FIRST)\n")
        f.write("=" * 75 + "\n\n")
        f.write(f"Started At:       {queue_state.get('start_time', 'N/A')}\n")
        f.write(f"Last Updated:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Overall Progress: {queue_state.get('completed_count', 0)} / {len(queue_state.get('tasks', []))} tasks completed\n\n")

        f.write("-" * 75 + "\n")
        f.write(f"{'Task':<10}{'Name':<35}{'Target':<10}{'Status':<15}{'Runtime':<10}\n")
        f.write("-" * 75 + "\n")
        for t in queue_state.get("tasks", []):
            f.write(f"{t['id']:<10}{t['name']:<35}{t['target']:<10}{t['status']:<15}{t.get('runtime', '--'):<10}\n")
        f.write("-" * 75 + "\n\n")

        if queue_state.get("current_task"):
            f.write(f"CURRENTLY RUNNING: {queue_state['current_task']}\n")
            f.write(f"Check live progress: results/{queue_state.get('current_progress_file', '')}\n")


def git_commit_and_push(task_id: str, task_name: str, result_files: list):
    print(f"\n[Git] Committing and pushing {task_id} results to GitHub...")
    try:
        add_files = " ".join(result_files)
        subprocess.run(f"git add {add_files}", shell=True, check=True, cwd=ROOT_DIR)
        commit_msg = f"Reproduce {task_id}: {task_name}"
        subprocess.run(f'git commit -m "{commit_msg}"', shell=True, check=True, cwd=ROOT_DIR)
        subprocess.run("git push origin main", shell=True, check=True, cwd=ROOT_DIR)
        print(f"[Git] Successfully pushed {task_id} to GitHub!\n")
    except Exception as e:
        print(f"[Git Warning] Git commit/push encountered: {e}")


def run_command_logged(cmd: list):
    print(f"\nExecuting: {' '.join(cmd)}")
    p = subprocess.Popen(cmd, cwd=ROOT_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in iter(p.stdout.readline, ""):
        print(line, end="")
    p.stdout.close()
    return_code = p.wait()
    return return_code == 0


def main():
    start_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    queue_tasks = [
        {
            "id": "Task 8",
            "name": "Clean Model + Clean + Equi Def",
            "target": "37.95%",
            "cmd": [
                sys.executable, "evaluate_clean_equi_defense.py",
                "--checkpoint", "checkpoints/clean_drn_d_22_cityscapes.pth",
                "--output-prefix", "baseline_drn_d_22_clean_equidefense",
                "--epsilon-defense", str(6.0 / 255.0),
                "--num-views", "7",
                "--batch-size", "2"
            ],
            "prefix": "baseline_drn_d_22_clean_equidefense",
            "status": "QUEUED"
        },
        {
            "id": "Task 10",
            "name": "Clean Model + PGD-50 + Equi Def",
            "target": "30.76%",
            "cmd": [
                sys.executable, "evaluate_equi_defense.py",
                "--checkpoint", "checkpoints/clean_drn_d_22_cityscapes.pth",
                "--output-prefix", "baseline_drn_d_22_pgd_equidefense",
                "--reverse-multiply", "1.5",
                "--num-views", "7",
                "--batch-size", "2"
            ],
            "prefix": "baseline_drn_d_22_pgd_equidefense",
            "status": "QUEUED"
        },
        {
            "id": "Task 6",
            "name": "Adv Model + BPDA + Equi Def",
            "target": "29.83%",
            "cmd": [
                sys.executable, "evaluate_bpda_defense.py",
                "--checkpoint", "checkpoints/advtrain_drn_d_22_cityscapes.pth.tar",
                "--output-prefix", "baseline_advtrain_drn_d_22_bpda_equidefense",
                "--batch-size", "2"
            ],
            "prefix": "baseline_advtrain_drn_d_22_bpda_equidefense",
            "status": "QUEUED"
        }
    ]

    queue_state = {
        "start_time": start_str,
        "completed_count": 0,
        "tasks": queue_tasks,
        "current_task": None,
        "current_progress_file": None
    }

    update_status(queue_state)

    for i, t in enumerate(queue_tasks):
        queue_state["current_task"] = f"{t['id']} ({t['name']})"
        queue_state["current_progress_file"] = f"{t['prefix']}_progress.txt"
        t["status"] = "RUNNING"
        update_status(queue_state)

        t_start = time.time()
        print(f"\n{'='*75}\n STARTING {t['id']}: {t['name']}\n{'='*75}")
        success = run_command_logged(t["cmd"])
        t_duration = time.time() - t_start
        mins = t_duration / 60.0

        if success:
            t["status"] = "COMPLETED"
            t["runtime"] = f"{mins:.1f}m"
            queue_state["completed_count"] += 1

            # Auto-git commit & push
            result_files = [
                f"results/{t['prefix']}_metrics.json",
                f"results/{t['prefix']}_summary.txt",
                f"results/{t['prefix']}_progress.txt"
            ]
            git_commit_and_push(t["id"], t["name"], result_files)
        else:
            t["status"] = "FAILED"
            t["runtime"] = f"{mins:.1f}m"

        update_status(queue_state)

    queue_state["current_task"] = None
    queue_state["current_progress_file"] = None
    update_status(queue_state)
    print("\n" + "=" * 75)
    print(" ALL QUEUE TASKS FINISHED! OVERNIGHT RUN COMPLETE! ")
    print("=" * 75)


if __name__ == "__main__":
    main()
