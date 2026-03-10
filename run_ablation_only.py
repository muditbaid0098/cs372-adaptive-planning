# Ablation study runner

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
MAX_WORKERS = 8

ABLATION_VARIANTS = {
    "ours_full": {"use_classifier": True, "use_adaptive_strategy": True,
                  "use_verifier": True, "use_failure_handler": True},
    "no_classifier": {"use_classifier": False, "use_adaptive_strategy": True,
                      "use_verifier": True, "use_failure_handler": True},
    "no_adaptive_strategy": {"use_classifier": True, "use_adaptive_strategy": False,
                             "use_verifier": True, "use_failure_handler": True},
    "no_verifier": {"use_classifier": True, "use_adaptive_strategy": True,
                    "use_verifier": False, "use_failure_handler": False},
    "no_failure_handler": {"use_classifier": True, "use_adaptive_strategy": True,
                           "use_verifier": True, "use_failure_handler": False},
    "decompose_only": {"use_classifier": False, "use_adaptive_strategy": False,
                       "use_verifier": False, "use_failure_handler": False},
}


def run_variant_on_task(task, variant_name, kwargs):
    """Run one ablation variant on one task."""
    # returns (variant_name, result_dict)
    reset_stats()
    start = time.time()
    try:
        result = run_adaptive_planner(task["initial"], task["goal"], **kwargs)
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
    out["method"] = variant_name
    out["wall_time"] = time.time() - start
    out["optimal_length"] = task.get("optimal_length", -1)

    status = "OK" if out.get("success") else "FAIL"
    print(f"  {variant_name:>25} | {task['id']:>12} ({task['difficulty']}): "
          f"{status} {out['wall_time']:.0f}s", flush=True)
    return variant_name, out


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
    print(f"Done. {len(tasks)} tasks, {len(ABLATION_VARIANTS)} variants.\n")

    # Build all (variant, task) jobs
    jobs = []
    for task in tasks:
        for vname, vkwargs in ABLATION_VARIANTS.items():
            jobs.append((task, vname, vkwargs))

    print(f"Running {len(jobs)} jobs with {MAX_WORKERS} workers...\n")

    results = {name: [] for name in ABLATION_VARIANTS}
    completed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_variant_on_task, t, vn, vk): (t, vn)
                   for t, vn, vk in jobs}
        for future in as_completed(futures):
            vname, out = future.result()
            results[vname].append(out)
            completed += 1
            if completed % 20 == 0:
                print(f"--- {completed}/{len(jobs)} jobs done ---", flush=True)

    # Sort
    for vname in results:
        results[vname].sort(key=lambda r: r["task_id"])

    # Save
    out_path = os.path.join(RESULTS_DIR, "ablation.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")

    # Summary
    print(f"\n{'Variant':<25} {'Easy':>8} {'Medium':>8} {'Hard':>8} {'Overall':>8}")
    print("=" * 60)
    for vname in ABLATION_VARIANTS:
        by_diff = {"easy": [], "medium": [], "hard": []}
        for r in results[vname]:
            by_diff[r["difficulty"]].append(r.get("success", False))
        rates = {}
        for diff in by_diff:
            rates[diff] = sum(by_diff[diff]) / len(by_diff[diff]) * 100 if by_diff[diff] else 0
        all_s = [r.get("success", False) for r in results[vname]]
        overall = sum(all_s) / len(all_s) * 100
        print(f"{vname:<25} {rates['easy']:>7.0f}% {rates['medium']:>7.0f}% {rates['hard']:>7.0f}% {overall:>7.0f}%")
