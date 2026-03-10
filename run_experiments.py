# Main experiment runner

import json
import time
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

sys.path.insert(0, os.path.dirname(__file__))

from src.blocksworld import generate_benchmark, bfs_optimal_plan
from src.baselines import flat_cot, flat_tot, react
from src.planner import run_adaptive_planner
from src.llm import reset_stats, get_stats

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
TRACES_DIR = os.path.join(os.path.dirname(__file__), "traces")
os.makedirs(TRACES_DIR, exist_ok=True)

# Number of tasks per difficulty
N_EASY = 20
N_MEDIUM = 20
N_HARD = 20
ATTEMPTS_PER_TASK = 3  # best of 3


def run_method(method_fn, initial, goal, method_name, **kwargs):
    """Run a method with retries."""
    for attempt in range(ATTEMPTS_PER_TASK):
        reset_stats()
        try:
            if method_name in ("ours", "ours_full", "no_classifier", "no_adaptive_strategy",
                               "no_verifier", "no_failure_handler", "decompose_only"):
                result = method_fn(initial, goal, **kwargs)
                stats = get_stats()
                return {
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
            else:
                result = method_fn(initial, goal)
                stats = get_stats()
                result["llm_calls"] = stats["calls"]
                result["total_tokens"] = stats["tokens"]
                return result
        except Exception as e:
            if attempt == ATTEMPTS_PER_TASK - 1:
                return {"success": False, "error": str(e), "total_steps": 0,
                        "llm_calls": 0, "total_tokens": 0}
            time.sleep(1)
    return {"success": False, "error": "all attempts failed", "total_steps": 0,
            "llm_calls": 0, "total_tokens": 0}


MAX_PARALLEL_TASKS = 8  # concurrent tasks; tune based on your API tier


def _run_single_task(task, methods):
    """Run all methods on one task."""
    task_id = task["id"]
    difficulty = task["difficulty"]
    initial = task["initial"]
    goal = task["goal"]
    task_results = []

    for method_name, method_config in methods.items():
        start = time.time()
        method_fn = method_config["fn"]
        kwargs = method_config.get("kwargs", {})

        result = run_method(method_fn, initial, goal, method_name, **kwargs)
        result["task_id"] = task_id
        result["difficulty"] = difficulty
        result["num_blocks"] = task["num_blocks"]
        result["method"] = method_name
        result["wall_time"] = time.time() - start
        result["optimal_length"] = task.get("optimal_length", -1)

        task_results.append((method_name, result))
        status = "OK" if result.get("success") else "FAIL"
        print(f"  {method_name} on {task_id} ({difficulty}): {status} ({result.get('wall_time', 0):.1f}s)")

    return task_results


def run_experiment(methods: dict, tasks: list, experiment_name: str, parallel: bool = True):
    """Run all methods on all tasks, optionally in parallel."""
    print("Computing optimal plan lengths...")
    for task in tasks:
        opt_len, opt_actions = bfs_optimal_plan(task["initial"], task["goal"])
        task["optimal_length"] = opt_len
    print(f"Done. Optimal lengths: {[t['optimal_length'] for t in tasks[:5]]}...")

    results = {name: [] for name in methods}
    total = len(tasks)

    if parallel and total > 1:
        n_workers = min(MAX_PARALLEL_TASKS, total)
        print(f"\nRunning {total} tasks with {n_workers} parallel workers...\n")
        completed = 0
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_run_single_task, task, methods): task for task in tasks}
            for future in as_completed(futures):
                completed += 1
                task_results = future.result()
                for method_name, result in task_results:
                    results[method_name].append(result)
                print(f"--- [{completed}/{total}] tasks complete ---")
    else:
        for i, task in enumerate(tasks):
            print(f"\n--- Task [{i+1}/{total}] ---")
            task_results = _run_single_task(task, methods)
            for method_name, result in task_results:
                results[method_name].append(result)

    # Sort results by task_id for consistent ordering
    for method_name in results:
        results[method_name].sort(key=lambda r: r["task_id"])

    # Save results
    output_path = os.path.join(RESULTS_DIR, f"{experiment_name}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")
    return results


def print_summary(results: dict):
    """Print a quick summary table."""
    print("\n" + "=" * 80)
    print(f"{'Method':<30} {'Easy':>8} {'Medium':>8} {'Hard':>8} {'Overall':>8}")
    print("=" * 80)

    for method_name, method_results in results.items():
        by_diff = {"easy": [], "medium": [], "hard": []}
        for r in method_results:
            by_diff[r["difficulty"]].append(r.get("success", False))

        rates = {}
        for diff in ["easy", "medium", "hard"]:
            if by_diff[diff]:
                rates[diff] = sum(by_diff[diff]) / len(by_diff[diff]) * 100
            else:
                rates[diff] = 0.0

        all_success = [r.get("success", False) for r in method_results]
        overall = sum(all_success) / len(all_success) * 100 if all_success else 0

        print(f"{method_name:<30} {rates['easy']:>7.1f}% {rates['medium']:>7.1f}% {rates['hard']:>7.1f}% {overall:>7.1f}%")

    print("=" * 80)


def run_main_experiment():
    """Main comparison: ours vs baselines."""
    print("Generating benchmark tasks...")
    tasks = generate_benchmark(easy=N_EASY, medium=N_MEDIUM, hard=N_HARD)
    print(f"Generated {len(tasks)} tasks.\n")

    methods = {
        "flat_cot": {"fn": flat_cot},
        "flat_tot": {"fn": flat_tot},
        "react": {"fn": react},
        "ours": {"fn": run_adaptive_planner, "kwargs": {
            "use_classifier": True,
            "use_adaptive_strategy": True,
            "use_verifier": True,
            "use_failure_handler": True,
        }},
    }

    results = run_experiment(methods, tasks, "main_comparison")
    print_summary(results)
    return results


def run_ablation_experiment():
    """Ablation: disable one component at a time."""
    print("\nRunning ablation study...")
    tasks = generate_benchmark(easy=N_EASY, medium=N_MEDIUM, hard=N_HARD)

    methods = {
        "ours_full": {"fn": run_adaptive_planner, "kwargs": {
            "use_classifier": True, "use_adaptive_strategy": True,
            "use_verifier": True, "use_failure_handler": True,
        }},
        "no_classifier": {"fn": run_adaptive_planner, "kwargs": {
            "use_classifier": False, "use_adaptive_strategy": True,
            "use_verifier": True, "use_failure_handler": True,
        }},
        "no_adaptive_strategy": {"fn": run_adaptive_planner, "kwargs": {
            "use_classifier": True, "use_adaptive_strategy": False,
            "use_verifier": True, "use_failure_handler": True,
        }},
        "no_verifier": {"fn": run_adaptive_planner, "kwargs": {
            "use_classifier": True, "use_adaptive_strategy": True,
            "use_verifier": False, "use_failure_handler": False,
        }},
        "no_failure_handler": {"fn": run_adaptive_planner, "kwargs": {
            "use_classifier": True, "use_adaptive_strategy": True,
            "use_verifier": True, "use_failure_handler": False,
        }},
        "decompose_only": {"fn": run_adaptive_planner, "kwargs": {
            "use_classifier": False, "use_adaptive_strategy": False,
            "use_verifier": False, "use_failure_handler": False,
        }},
    }

    results = run_experiment(methods, tasks, "ablation")
    print_summary(results)
    return results


def run_scaling_experiment():
    """Scaling: test on harder 9-10 block tasks."""
    print("\nRunning scaling experiment (9-10 blocks)...")
    import random
    from src.blocksworld import generate_task
    random.seed(42)

    tasks = []
    for i in range(10):
        init, goal = generate_task(9, seed=300 + i)
        tasks.append({"id": f"extra_hard_9_{i}", "difficulty": "extra_hard",
                       "num_blocks": 9, "initial": init, "goal": goal})
    for i in range(10):
        init, goal = generate_task(10, seed=400 + i)
        tasks.append({"id": f"extra_hard_10_{i}", "difficulty": "extra_hard",
                       "num_blocks": 10, "initial": init, "goal": goal})

    methods = {
        "flat_cot": {"fn": flat_cot},
        "react": {"fn": react},
        "ours": {"fn": run_adaptive_planner, "kwargs": {
            "use_classifier": True, "use_adaptive_strategy": True,
            "use_verifier": True, "use_failure_handler": True,
        }},
    }

    results = run_experiment(methods, tasks, "scaling")
    print_summary(results)
    return results


def run_model_comparison():
    """Compare with GPT-4o-mini."""
    import src.llm as llm_module
    original_model = llm_module.DEFAULT_MODEL
    llm_module.DEFAULT_MODEL = "gpt-4o-mini"

    print(f"\nRunning model comparison with {llm_module.DEFAULT_MODEL}...")
    tasks = generate_benchmark(easy=10, medium=10, hard=10, seed=42)

    methods = {
        "flat_cot_mini": {"fn": flat_cot},
        "react_mini": {"fn": react},
        "ours_mini": {"fn": run_adaptive_planner, "kwargs": {
            "use_classifier": True, "use_adaptive_strategy": True,
            "use_verifier": True, "use_failure_handler": True,
        }},
    }

    results = run_experiment(methods, tasks, "model_comparison")
    print_summary(results)

    llm_module.DEFAULT_MODEL = original_model
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["main", "ablation", "scaling", "model", "all"],
                        default="all")
    parser.add_argument("--small", action="store_true", help="Use smaller task set for testing")
    args = parser.parse_args()

    if args.small:
        N_EASY = 5
        N_MEDIUM = 5
        N_HARD = 5

    if args.experiment in ["main", "all"]:
        run_main_experiment()
    if args.experiment in ["ablation", "all"]:
        run_ablation_experiment()
    if args.experiment in ["scaling", "all"]:
        run_scaling_experiment()
    if args.experiment in ["model"]:
        run_model_comparison()
