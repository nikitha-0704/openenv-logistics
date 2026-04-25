"""Tiny rollout + plotting harness for the Round 2 mini-blog / video.

What it does
------------
1. For each task in {easy, medium, hard}, run N seeds of:
     * scripted-baseline (uses `test_local.py` action lists, no LLM)
     * single-agent LLM (calls `inference.run_task` if env vars set)
     * multi-agent LLM  (calls `inference_multi.run_task` if env vars set)
2. Reads the grader score from the env's `/grader` endpoint after each run.
3. Writes a JSON summary and a grouped bar chart PNG.

Usage
-----
    # Make sure the env server is up:
    python -m server.app   # in another shell, on http://localhost:7860

    # Then:
    python -m training.collect_and_plot --seeds 3 --out docs/plots

The script keeps LLM calls behind a flag (`--llm`) so reviewers can reproduce
the scripted bars with no API key.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OPENENV_BASE_URL = (
    os.environ.get("OPENENV_BASE_URL")
    or os.environ.get("ENV_BASE_URL")
    or "http://localhost:7860"
)


def _scripted_actions(task: str) -> list[dict]:
    """Mirror of `test_local.py` for a deterministic, no-LLM baseline."""
    if task == "easy":
        return [
            {"action_type": "check_network"},
            {"action_type": "load_truck", "truck_id": "T101", "warehouse": "North", "amount": 20},
            {"action_type": "route_truck", "truck_id": "T101", "route_id": "North_to_South"},
            {"action_type": "wait", "hours": 5},
            {"action_type": "load_truck", "truck_id": "T101", "warehouse": "South", "amount": 20},
            {"action_type": "route_truck", "truck_id": "T101", "route_id": "South_to_East"},
            {"action_type": "wait", "hours": 8},
        ]
    if task == "medium":
        return [
            {"action_type": "check_network"},
            {"action_type": "load_truck", "truck_id": "T101", "warehouse": "North", "amount": 50},
            {"action_type": "route_truck", "truck_id": "T101", "route_id": "North_to_East"},
            {"action_type": "wait", "hours": 10},
        ]
    if task == "hard":
        return [
            {"action_type": "check_network"},
            {"action_type": "load_truck", "truck_id": "T101", "warehouse": "North", "amount": 40},
            {"action_type": "route_truck", "truck_id": "T101", "route_id": "North_to_East"},
            {"action_type": "route_truck", "truck_id": "T102", "route_id": "South_to_North"},
            {"action_type": "wait", "hours": 5},
            {"action_type": "load_truck", "truck_id": "T102", "warehouse": "North", "amount": 10},
            {"action_type": "route_truck", "truck_id": "T102", "route_id": "North_to_South"},
            {"action_type": "wait", "hours": 10},
        ]
    return []


def _run_scripted(task: str) -> float:
    requests.post(f"{OPENENV_BASE_URL}/reset", json={"task_level": task}, timeout=30)
    for action in _scripted_actions(task):
        try:
            r = requests.post(f"{OPENENV_BASE_URL}/step", json=action, timeout=30)
            if r.status_code == 200 and r.json().get("done"):
                break
        except requests.RequestException:
            break
    return float(requests.get(f"{OPENENV_BASE_URL}/grader", timeout=30).json().get("score", 0.0))


def _run_single_llm(task: str) -> float:
    from inference import run_task as run_single

    return float(run_single(task))


def _run_multi_llm(task: str) -> float:
    from inference_multi import run_task as run_multi

    return float(run_multi(task))


def _mean(xs: list[float]) -> float:
    return round(sum(xs) / max(1, len(xs)), 4)


def _stderr(xs: list[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return round((var / len(xs)) ** 0.5, 4)


_RUN_LABEL = {
    "scripted": "Scripted optimum (no LLM)",
    "single_agent_llm": "Single-agent LLM",
    "multi_agent_llm": "Hierarchical multi-agent LLM",
}


def _plot(summary: dict, out_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; install with `pip install matplotlib` to save plot.", flush=True)
        return

    tasks = ["easy", "medium", "hard"]
    runs = list(summary["runs"].keys())
    n_runs = max(1, len(runs))
    bar_width = 0.8 / n_runs
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for i, run_name in enumerate(runs):
        per_task = summary["runs"][run_name]
        means = [_mean(per_task.get(t, [])) for t in tasks]
        errs = [_stderr(per_task.get(t, [])) for t in tasks]
        xs = [j + i * bar_width for j in range(len(tasks))]
        bars = ax.bar(
            xs, means, width=bar_width,
            label=_RUN_LABEL.get(run_name, run_name),
            yerr=errs if any(e > 0 for e in errs) else None,
            capsize=3,
        )
        for bar, mean in zip(bars, means):
            ax.annotate(
                f"{mean:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, mean),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks([j + bar_width * (n_runs - 1) / 2 for j in range(len(tasks))])
    ax.set_xticklabels([t.capitalize() for t in tasks])
    ax.set_xlabel("Task difficulty")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Terminal grader score (0–1, open-interval)")
    seed_label = f"seeds = {summary.get('seeds', '?')} per (run × task); error bars = SEM"
    ax.set_title(
        f"Global Logistics Resolver — agent comparison\n{seed_label}",
        fontsize=11,
    )
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=min(3, n_runs), frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"wrote {out_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=1, help="Repetitions per task")
    parser.add_argument("--llm", action="store_true", help="Also run the single- and multi-agent LLM rollouts")
    parser.add_argument(
        "--out",
        type=str,
        default=str(REPO_ROOT / "docs" / "plots"),
        help="Output directory for JSON + PNG",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs: dict[str, dict[str, list[float]]] = {"scripted": {"easy": [], "medium": [], "hard": []}}
    if args.llm:
        runs["single_agent_llm"] = {"easy": [], "medium": [], "hard": []}
        runs["multi_agent_llm"] = {"easy": [], "medium": [], "hard": []}

    for task in ("easy", "medium", "hard"):
        for _ in range(args.seeds):
            runs["scripted"][task].append(_run_scripted(task))
            time.sleep(0.1)
            if args.llm:
                runs["single_agent_llm"][task].append(_run_single_llm(task))
                time.sleep(0.1)
                runs["multi_agent_llm"][task].append(_run_multi_llm(task))
                time.sleep(0.1)

    summary = {"env": OPENENV_BASE_URL, "seeds": args.seeds, "runs": runs}
    summary_path = out_dir / "rollout_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {summary_path}", flush=True)

    _plot(summary, out_dir / "baseline_vs_multi.png")


if __name__ == "__main__":
    main()
