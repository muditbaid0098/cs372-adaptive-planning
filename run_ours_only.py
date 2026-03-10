# Run just our method (reuses baseline results from previous runs)

import json
import time
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))

from src.blocksworld import generate_benchmark, bfs_optimal_plan
from src.planner import run_adaptive_planner
from src.llm import reset_stats, get_stats

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
MAX_WORKERS = 4


def run_ours_on_task(task):
    """Run our method on one task."""
    reset_stats()
    start = time.time()
    try:
        result = run_adaptive_planner(
            task["initial"], task["goal"],
            use_classifier=True, use_adaptive_strategy=True,
            use_verifier=True, use_failure_handler=True,
        )
        stats = get_stats()
        out = {
            "success": result.success,
            "total_steps": result.total_steps,
            "rollbacks": result.rollbacks,
            "strategy_switches": result.strategy_switches,
            "recovery_attempts": result.recovery_attempts,
            "llm_calls": stats["calls"],
            "total_tokens": stats["tokens"],
            "elapsed_time": result.elapsed_time,
            "failure_reason": result.failure_reason,
            "recovery_trace": result.recovery_trace,
        }
    except Exception as e:
        out = {"success": False, "error": str(e), "total_steps": 0,
               "llm_calls": 0, "total_tokens": 0}

    out["task_id"] = task["id"]
    out["difficulty"] = task["difficulty"]
    out["num_blocks"] = task["num_blocks"]
    out["method"] = "ours"
    out["wall_time"] = time.time() - start
    out["optimal_length"] = task.get("optimal_length", -1)

    status = "OK" if out.get("success") else "FAIL"
    print(f"  {task['id']} ({task['difficulty']}, {task['num_blocks']}b): "
          f"{status} steps={out.get('total_steps',0)} "
          f"recovery={out.get('recovery_attempts',0)} "
          f"time={out['wall_time']:.1f}s", flush=True)
    return out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", action="store_true")
    args = parser.parse_args()

    n = 5 if args.small else 20
    tasks = generate_benchmark(easy=n, medium=n, hard=n, seed=42)

    # Compute optimal lengths
    print("Computing optimal plan lengths...")
    for task in tasks:
        opt_len, _ = bfs_optimal_plan(task["initial"], task["goal"])
        task["optimal_length"] = opt_len
    print(f"Done. {len(tasks)} tasks.\n")

    # Run ours in parallel
    print(f"Running 'ours' on {len(tasks)} tasks with {MAX_WORKERS} workers...\n")
    ours_results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_ours_on_task, t): t for t in tasks}
        for future in as_completed(futures):
            ours_results.append(future.result())

    ours_results.sort(key=lambda r: r["task_id"])

    # Safety: don't overwrite if most tasks failed (likely rate limit)
    success_rate = sum(1 for r in ours_results if r.get("success")) / len(ours_results)
    if success_rate < 0.1:
        out_path = os.path.join(RESULTS_DIR, "ours_failed_run.json")
        with open(out_path, "w") as f:
            json.dump(ours_results, f, indent=2, default=str)
        print(f"\nWARNING: Very low success rate ({success_rate:.0%}), likely rate limited.")
        print(f"Saved to {out_path} instead of overwriting main_comparison.json")
    else:
        # Load baseline results if they exist
        main_path = os.path.join(RESULTS_DIR, "main_comparison.json")
        if os.path.exists(main_path):
            print("\nLoading existing baseline results...")
            with open(main_path) as f:
                existing = json.load(f)
            existing["ours"] = ours_results
            combined = existing
        else:
            combined = {"ours": ours_results}

        out_path = os.path.join(RESULTS_DIR, "main_comparison.json")
        with open(out_path, "w") as f:
            json.dump(combined, f, indent=2, default=str)
        print(f"\nSaved to {out_path}")

    # Summary
    by_diff = {"easy": [], "medium": [], "hard": []}
    for r in ours_results:
        by_diff[r["difficulty"]].append(r.get("success", False))

    print(f"\n{'Difficulty':<10} {'Rate':>8}")
    print("-" * 20)
    for diff in ["easy", "medium", "hard"]:
        if by_diff[diff]:
            rate = sum(by_diff[diff]) / len(by_diff[diff]) * 100
            print(f"{diff:<10} {rate:>7.0f}%")
    all_s = [r.get("success", False) for r in ours_results]
    print(f"{'overall':<10} {sum(all_s)/len(all_s)*100:>7.0f}%")
