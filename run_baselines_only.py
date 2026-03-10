# Run just the baselines (flat CoT, flat ToT, ReAct)

import json
import time
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))

from src.blocksworld import generate_benchmark, bfs_optimal_plan
from src.baselines import flat_cot, flat_tot, react
from src.llm import reset_stats, get_stats

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
MAX_WORKERS = 8

BASELINES = {
    "flat_cot": flat_cot,
    "flat_tot": flat_tot,
    "react": react,
}


def run_baseline_on_task(task, method_name, method_fn):
    """Run one baseline on one task, return (method_name, result)."""
    reset_stats()
    start = time.time()
    try:
        result = method_fn(task["initial"], task["goal"])
        stats = get_stats()
        result["llm_calls"] = stats["calls"]
        result["total_tokens"] = stats["tokens"]
    except Exception as e:
        result = {"success": False, "error": str(e), "total_steps": 0,
                  "llm_calls": 0, "total_tokens": 0}

    result["task_id"] = task["id"]
    result["difficulty"] = task["difficulty"]
    result["num_blocks"] = task["num_blocks"]
    result["method"] = method_name
    result["wall_time"] = time.time() - start
    result["optimal_length"] = task.get("optimal_length", -1)

    status = "OK" if result.get("success") else "FAIL"
    print(f"  {method_name:>10} | {task['id']:>12} ({task['difficulty']}): "
          f"{status} {result['wall_time']:.0f}s", flush=True)
    return method_name, result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", action="store_true")
    args = parser.parse_args()

    n = 5 if args.small else 20
    tasks = generate_benchmark(easy=n, medium=n, hard=n, seed=42)

    print("Computing optimal plan lengths...")
    for task in tasks:
        opt_len, _ = bfs_optimal_plan(task["initial"], task["goal"])
        task["optimal_length"] = opt_len
    print(f"Done. {len(tasks)} tasks, {len(BASELINES)} baselines.\n")

    # Build jobs: 3 baselines x 60 tasks = 180 jobs
    jobs = []
    for task in tasks:
        for mname, mfn in BASELINES.items():
            jobs.append((task, mname, mfn))

    print(f"Running {len(jobs)} jobs with {MAX_WORKERS} workers...\n")

    results = {name: [] for name in BASELINES}
    completed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_baseline_on_task, t, mn, mf): (t, mn)
                   for t, mn, mf in jobs}
        for future in as_completed(futures):
            mname, out = future.result()
            results[mname].append(out)
            completed += 1
            if completed % 15 == 0:
                print(f"--- {completed}/{len(jobs)} jobs done ---", flush=True)

    # Sort
    for mname in results:
        results[mname].sort(key=lambda r: r["task_id"])

    # Merge with existing ours results
    ours_path = os.path.join(RESULTS_DIR, "ours_60_backup.json")
    main_path = os.path.join(RESULTS_DIR, "main_comparison.json")

    if os.path.exists(ours_path):
        print("\nLoading existing ours results...")
        with open(ours_path) as f:
            ours_results = json.load(f)
        results["ours"] = ours_results
    elif os.path.exists(main_path):
        with open(main_path) as f:
            existing = json.load(f)
        if "ours" in existing and len(existing["ours"]) == 3 * n:
            results["ours"] = existing["ours"]

    # Save
    with open(main_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {main_path}")

    # Summary
    print(f"\n{'Method':<15} {'Easy':>8} {'Medium':>8} {'Hard':>8} {'Overall':>8}")
    print("=" * 50)
    for mname in results:
        by_diff = {"easy": [], "medium": [], "hard": []}
        for r in results[mname]:
            by_diff[r["difficulty"]].append(r.get("success", False))
        rates = {}
        for diff in by_diff:
            rates[diff] = sum(by_diff[diff]) / len(by_diff[diff]) * 100 if by_diff[diff] else 0
        all_s = [r.get("success", False) for r in results[mname]]
        overall = sum(all_s) / len(all_s) * 100 if all_s else 0
        print(f"{mname:<15} {rates['easy']:>7.0f}% {rates['medium']:>7.0f}% {rates['hard']:>7.0f}% {overall:>7.0f}%")
