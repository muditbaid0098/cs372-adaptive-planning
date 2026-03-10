# Analyze results and spit out LaTeX tables

import json
import os
import sys
import math
import numpy as np
from scipy import stats as scipy_stats

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def load_results(filename):
    path = os.path.join(RESULTS_DIR, filename)
    with open(path) as f:
        return json.load(f)


### Stats helpers ###

def wilson_ci(successes, n, alpha=0.05):
    """Wilson score CI for a proportion."""
    if n == 0:
        return 0, 0, 0
    p_hat = successes / n
    z = scipy_stats.norm.ppf(1 - alpha / 2)
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    return p_hat, max(0, center - margin), min(1, center + margin)


def mcnemar_test(results_a, results_b, task_ids):
    """McNemar's test for paired outcomes, returns p-value."""
    a_map = {r["task_id"]: r.get("success", False) for r in results_a}
    b_map = {r["task_id"]: r.get("success", False) for r in results_b}
    # b=A success & B fail, c=A fail & B success
    b_count = sum(1 for t in task_ids if a_map.get(t) and not b_map.get(t))
    c_count = sum(1 for t in task_ids if not a_map.get(t) and b_map.get(t))
    if b_count + c_count == 0:
        return 1.0
    # Exact binomial test (better than chi-square for small counts)
    result = scipy_stats.binomtest(b_count, b_count + c_count, 0.5)
    return result.pvalue


### Core metrics ###

def compute_metrics(method_results):
    """Compute success rates, avg steps, etc."""
    by_diff = {"easy": [], "medium": [], "hard": [], "extra_hard": []}
    all_steps = []
    all_times = []
    all_calls = []
    all_tokens = []

    for r in method_results:
        diff = r.get("difficulty", "easy")
        if diff not in by_diff:
            by_diff[diff] = []
        by_diff[diff].append(r.get("success", False))
        if r.get("success"):
            all_steps.append(r.get("total_steps", 0))
            all_times.append(r.get("wall_time", 0))
        all_calls.append(r.get("llm_calls", 0))
        all_tokens.append(r.get("total_tokens", 0))

    metrics = {}
    for diff in by_diff:
        n = len(by_diff[diff])
        s = sum(by_diff[diff])
        if n > 0:
            rate, ci_lo, ci_hi = wilson_ci(s, n)
            metrics[f"{diff}_rate"] = rate * 100
            metrics[f"{diff}_ci_lo"] = ci_lo * 100
            metrics[f"{diff}_ci_hi"] = ci_hi * 100
            metrics[f"{diff}_n"] = n
            metrics[f"{diff}_successes"] = s

    total = sum(len(v) for v in by_diff.values())
    total_success = sum(sum(v) for v in by_diff.values())
    if total > 0:
        rate, ci_lo, ci_hi = wilson_ci(total_success, total)
        metrics["overall_rate"] = rate * 100
        metrics["overall_ci_lo"] = ci_lo * 100
        metrics["overall_ci_hi"] = ci_hi * 100
    else:
        metrics["overall_rate"] = 0

    metrics["avg_steps"] = np.mean(all_steps) if all_steps else 0
    metrics["avg_time"] = np.mean(all_times) if all_times else 0
    metrics["avg_calls"] = np.mean(all_calls) if all_calls else 0
    metrics["avg_tokens"] = np.mean(all_tokens) if all_tokens else 0
    metrics["total_successes"] = total_success
    metrics["total_tasks"] = total

    return metrics


def format_rate_ci(rate, ci_lo, ci_hi):
    """Format for LaTeX."""
    return f"{rate:.0f}\\% \\tiny{{({ci_lo:.0f}-{ci_hi:.0f})}}"


### LaTeX tables ###

def print_main_table(results):
    """LaTeX table for main comparison."""
    print("\n% Main comparison table")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{lcccc}")
    print("\\toprule")
    print("\\textbf{Method} & \\textbf{Easy} & \\textbf{Medium} & \\textbf{Hard} & \\textbf{Overall} \\\\")
    print("\\midrule")

    method_display = {
        "flat_cot": "Flat CoT",
        "flat_tot": "Flat ToT",
        "react": "ReAct",
        "ours": "Ours (full)",
    }

    for method_name in ["flat_cot", "flat_tot", "react", "ours"]:
        if method_name not in results:
            continue
        m = compute_metrics(results[method_name])
        display = method_display.get(method_name, method_name)
        if method_name == "ours":
            display = "\\textbf{" + display + "}"
            print(f"{display} & \\textbf{{{m['easy_rate']:.0f}\\%}} & \\textbf{{{m['medium_rate']:.0f}\\%}} & \\textbf{{{m['hard_rate']:.0f}\\%}} & \\textbf{{{m['overall_rate']:.0f}\\%}} \\\\")
        else:
            print(f"{display} & {m['easy_rate']:.0f}\\% & {m['medium_rate']:.0f}\\% & {m['hard_rate']:.0f}\\% & {m['overall_rate']:.0f}\\% \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{Task success rate (\\%) on BlocksWorld by difficulty level.}")
    print("\\label{tab:main_results}")
    print("\\end{table}")


def print_ablation_table(results):
    """LaTeX table for ablation."""
    print("\n% Ablation table")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{lcccc}")
    print("\\toprule")
    print("\\textbf{Variant} & \\textbf{Easy} & \\textbf{Medium} & \\textbf{Hard} & \\textbf{Overall} \\\\")
    print("\\midrule")

    display_names = {
        "ours_full": "Full system",
        "no_classifier": "$-$ Classifier",
        "no_adaptive_strategy": "$-$ Adaptive strategy",
        "no_verifier": "$-$ Verifier \\& Handler",
        "no_failure_handler": "$-$ Failure handler",
        "decompose_only": "Decompose only",
    }

    for method_name in ["ours_full", "no_classifier", "no_adaptive_strategy",
                         "no_verifier", "no_failure_handler", "decompose_only"]:
        if method_name not in results:
            continue
        m = compute_metrics(results[method_name])
        display = display_names.get(method_name, method_name)
        print(f"{display} & {m['easy_rate']:.0f}\\% & {m['medium_rate']:.0f}\\% & {m['hard_rate']:.0f}\\% & {m['overall_rate']:.0f}\\% \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{Ablation study. Each row removes one component from the full system.}")
    print("\\label{tab:ablation}")
    print("\\end{table}")


### Significance tests ###

def print_significance_tests(results):
    """Pairwise significance tests vs baselines."""
    if "ours" not in results:
        return

    task_ids = [r["task_id"] for r in results["ours"]]
    print("\n% Pairwise significance tests (McNemar's, two-sided)")
    print(f"{'Comparison':<35} {'p-value':>10} {'Significant':>12}")
    print("-" * 60)

    for baseline in ["flat_cot", "flat_tot", "react"]:
        if baseline not in results:
            continue
        p = mcnemar_test(results["ours"], results[baseline], task_ids)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"Ours vs {baseline:<25} {p:>10.4f} {sig:>12}")

    # Per-difficulty
    for diff in ["easy", "medium", "hard"]:
        diff_tasks = [r["task_id"] for r in results["ours"] if r["difficulty"] == diff]
        if not diff_tasks:
            continue
        for baseline in ["flat_cot", "react"]:
            if baseline not in results:
                continue
            diff_a = [r for r in results["ours"] if r["difficulty"] == diff]
            diff_b = [r for r in results[baseline] if r["difficulty"] == diff]
            p = mcnemar_test(diff_a, diff_b, diff_tasks)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            print(f"  {diff}: Ours vs {baseline:<20} {p:>10.4f} {sig:>12}")


### Efficiency ###

def print_efficiency_table(results):
    """LLM cost comparison table."""
    print("\n% Efficiency analysis")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{lccccc}")
    print("\\toprule")
    print("\\textbf{Method} & \\textbf{Avg Calls} & \\textbf{Avg Tokens} & \\textbf{Success \\%} & \\textbf{Tokens/Success} & \\textbf{Avg Time (s)} \\\\")
    print("\\midrule")

    method_display = {
        "flat_cot": "Flat CoT", "flat_tot": "Flat ToT",
        "react": "ReAct", "ours": "Ours (full)",
    }

    for method_name in ["flat_cot", "flat_tot", "react", "ours"]:
        if method_name not in results:
            continue
        m = compute_metrics(results[method_name])
        display = method_display.get(method_name, method_name)
        total_tokens = sum(r.get("total_tokens", 0) for r in results[method_name])
        n_success = m["total_successes"]
        tps = total_tokens / n_success if n_success > 0 else float('inf')
        tps_str = f"{tps:.0f}" if tps < 1e6 else "$\\infty$"
        print(f"{display} & {m['avg_calls']:.1f} & {m['avg_tokens']:.0f} & {m['overall_rate']:.0f}\\% & {tps_str} & {m['avg_time']:.1f} \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{Computational efficiency comparison. Tokens/Success = total tokens used divided by number of successful tasks.}")
    print("\\label{tab:efficiency}")
    print("\\end{table}")


### Plan optimality ###

def print_optimality_analysis(results):
    """How close are plans to BFS-optimal?"""
    print("\n% Plan optimality analysis")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{lcccc}")
    print("\\toprule")
    print("\\textbf{Method} & \\textbf{Easy Ratio} & \\textbf{Medium Ratio} & \\textbf{Hard Ratio} & \\textbf{Overall Ratio} \\\\")
    print("\\midrule")

    method_display = {
        "flat_cot": "Flat CoT", "flat_tot": "Flat ToT",
        "react": "ReAct", "ours": "Ours (full)",
    }

    for method_name in ["flat_cot", "flat_tot", "react", "ours"]:
        if method_name not in results:
            continue
        display = method_display.get(method_name, method_name)

        ratios_by_diff = {"easy": [], "medium": [], "hard": []}
        all_ratios = []
        for r in results[method_name]:
            if r.get("success") and r.get("optimal_length", -1) > 0:
                ratio = r["total_steps"] / r["optimal_length"]
                diff = r["difficulty"]
                if diff in ratios_by_diff:
                    ratios_by_diff[diff].append(ratio)
                all_ratios.append(ratio)

        parts = []
        for diff in ["easy", "medium", "hard"]:
            if ratios_by_diff[diff]:
                parts.append(f"{np.mean(ratios_by_diff[diff]):.2f}x")
            else:
                parts.append("--")
        overall = f"{np.mean(all_ratios):.2f}x" if all_ratios else "--"
        print(f"{display} & {parts[0]} & {parts[1]} & {parts[2]} & {overall} \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{Plan length ratio (actual / optimal). Lower is better; 1.00x = optimal.}")
    print("\\label{tab:optimality}")
    print("\\end{table}")

    # Also print raw optimal lengths for context
    if "ours" in results:
        for diff in ["easy", "medium", "hard"]:
            opts = [r["optimal_length"] for r in results["ours"]
                    if r["difficulty"] == diff and r.get("optimal_length", -1) > 0]
            if opts:
                print(f"% Optimal plan lengths ({diff}): mean={np.mean(opts):.1f}, range={min(opts)}-{max(opts)}")


### Recovery analysis ###

def print_recovery_analysis(results):
    """How often does recovery kick in and help?"""
    if "ours" not in results and "ours_full" not in results:
        return

    method_key = "ours" if "ours" in results else "ours_full"
    method_results = results[method_key]

    total = len(method_results)
    successes = sum(1 for r in method_results if r.get("success"))
    recovery_attempts = sum(r.get("recovery_attempts", 0) for r in method_results)
    rollbacks = sum(r.get("rollbacks", 0) for r in method_results)
    strategy_switches = sum(r.get("strategy_switches", 0) for r in method_results)

    tasks_with_recovery = sum(1 for r in method_results if r.get("recovery_attempts", 0) > 0)
    recovery_successes = sum(1 for r in method_results
                            if r.get("recovery_attempts", 0) > 0 and r.get("success"))

    # First-try vs recovery success
    first_try_success = sum(1 for r in method_results
                           if r.get("success") and r.get("recovery_attempts", 0) == 0)

    print("\n% Recovery analysis")
    print(f"Total tasks: {total}")
    print(f"Total successes: {successes} ({successes/total*100:.1f}\\%)")
    print(f"First-try successes: {first_try_success}")
    print(f"Tasks needing recovery: {tasks_with_recovery}")
    if tasks_with_recovery > 0:
        print(f"Recovery success rate: {recovery_successes}/{tasks_with_recovery} ({recovery_successes/tasks_with_recovery*100:.1f}\\%)")
    print(f"Total strategy switches: {strategy_switches}")
    print(f"Total rollbacks: {rollbacks}")

    # Recovery by difficulty
    for diff in ["easy", "medium", "hard"]:
        diff_results = [r for r in method_results if r["difficulty"] == diff]
        diff_recovery = [r for r in diff_results if r.get("recovery_attempts", 0) > 0]
        diff_recovery_success = sum(1 for r in diff_recovery if r.get("success"))
        if diff_recovery:
            print(f"  {diff}: {len(diff_recovery)} tasks needed recovery, {diff_recovery_success} recovered ({diff_recovery_success/len(diff_recovery)*100:.0f}\\%)")

    # Analyze recovery traces if available
    all_traces = []
    for r in method_results:
        for t in r.get("recovery_trace", []):
            t["task_success"] = r.get("success", False)
            t["difficulty"] = r.get("difficulty", "unknown")
            all_traces.append(t)

    if all_traces:
        print(f"\n% Recovery trace analysis ({len(all_traces)} recovery events)")
        by_action = {}
        for t in all_traces:
            action = t.get("action_taken", "unknown")
            if action not in by_action:
                by_action[action] = {"total": 0, "verified": 0}
            by_action[action]["total"] += 1
            if t.get("recovery_verified"):
                by_action[action]["verified"] += 1
        for action, counts in sorted(by_action.items()):
            print(f"  {action}: {counts['verified']}/{counts['total']} verified successful")


### Failure modes ###

def analyze_failure_modes(results):
    """What kinds of failures happen and where."""
    if "ours" not in results and "ours_full" not in results:
        return

    method_key = "ours" if "ours" in results else "ours_full"
    failures = [r for r in results[method_key] if not r.get("success")]

    if not failures:
        print("\n% No failures to analyze!")
        return

    def categorize(reason):
        if reason is None:
            return "Unknown"
        reason_lower = reason.lower()
        if "not on" in reason_lower or "ordering" in reason_lower:
            return "Ordering error"
        elif "not clear" in reason_lower or "already holding" in reason_lower:
            return "Precondition violation"
        elif "goal not reached" in reason_lower:
            return "Incomplete plan"
        elif "decomposition" in reason_lower:
            return "Decomposition failure"
        elif "max steps" in reason_lower:
            return "Timeout"
        elif "no actions" in reason_lower:
            return "Empty plan"
        else:
            return "Other"

    reasons = {}
    reasons_by_diff = {}
    for f in failures:
        cat = categorize(f.get("failure_reason"))
        reasons[cat] = reasons.get(cat, 0) + 1
        diff = f.get("difficulty", "unknown")
        if diff not in reasons_by_diff:
            reasons_by_diff[diff] = {}
        reasons_by_diff[diff][cat] = reasons_by_diff[diff].get(cat, 0) + 1

    print(f"\n% Failure mode analysis ({len(failures)} failures)")
    for cat, count in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count} ({count/len(failures)*100:.1f}\\%)")

    # By difficulty
    for diff in ["easy", "medium", "hard"]:
        if diff in reasons_by_diff:
            diff_fails = sum(reasons_by_diff[diff].values())
            print(f"\n% {diff.capitalize()} failures ({diff_fails}):")
            for cat, count in sorted(reasons_by_diff[diff].items(), key=lambda x: -x[1]):
                print(f"    {cat}: {count}")

    # Failure by num_blocks
    print("\n% Failure rate by num_blocks:")
    blocks_counts = {}
    for r in results[method_key]:
        nb = r.get("num_blocks", 0)
        if nb not in blocks_counts:
            blocks_counts[nb] = {"total": 0, "fail": 0}
        blocks_counts[nb]["total"] += 1
        if not r.get("success"):
            blocks_counts[nb]["fail"] += 1
    for nb in sorted(blocks_counts):
        c = blocks_counts[nb]
        print(f"  {nb} blocks: {c['fail']}/{c['total']} failed ({c['fail']/c['total']*100:.0f}\\%)")


### Cross-method analysis ###

def print_cross_method_analysis(results):
    """Which tasks does only ours solve? Which do all methods fail?"""
    if "ours" not in results:
        return

    task_ids = [r["task_id"] for r in results["ours"]]
    method_success = {}
    for method_name, method_results in results.items():
        method_success[method_name] = {r["task_id"]: r.get("success", False) for r in method_results}

    baselines = [m for m in ["flat_cot", "flat_tot", "react"] if m in results]

    # Tasks where ours succeeds but all baselines fail
    ours_unique = []
    # Tasks where all methods fail
    all_fail = []
    # Tasks where a baseline succeeds but ours fails
    ours_fails_baseline_succeeds = []

    for tid in task_ids:
        ours_ok = method_success["ours"].get(tid, False)
        baseline_any = any(method_success[b].get(tid, False) for b in baselines)
        baseline_all_fail = not baseline_any

        if ours_ok and baseline_all_fail:
            ours_unique.append(tid)
        if not ours_ok and baseline_all_fail:
            all_fail.append(tid)
        if not ours_ok and baseline_any:
            ours_fails_baseline_succeeds.append(tid)

    print(f"\n% Cross-method analysis")
    print(f"Tasks only ours solves (baselines all fail): {len(ours_unique)}")
    print(f"Tasks all methods fail: {len(all_fail)}")
    print(f"Tasks a baseline solves but ours fails: {len(ours_fails_baseline_succeeds)}")

    # Characterize the "all fail" tasks
    if all_fail:
        all_fail_details = [r for r in results["ours"] if r["task_id"] in all_fail]
        blocks = [r["num_blocks"] for r in all_fail_details]
        print(f"  All-fail tasks avg blocks: {np.mean(blocks):.1f}, range: {min(blocks)}-{max(blocks)}")


### Case studies ###

def find_interesting_cases(results):
    """Find good examples for the case study section."""
    if "ours" not in results:
        return

    print("\n% === Interesting cases for qualitative analysis ===")

    ours = results["ours"]
    baselines = {m: {r["task_id"]: r for r in results[m]}
                 for m in ["flat_cot", "flat_tot", "react"] if m in results}

    # Case 1: Recovery success stories (hard tasks)
    recovery_successes = [r for r in ours
                          if r.get("success") and r.get("recovery_attempts", 0) > 0
                          and r.get("difficulty") in ("medium", "hard")]
    print(f"\n% Recovery success stories: {len(recovery_successes)}")
    for r in recovery_successes[:3]:
        print(f"  {r['task_id']}: {r['num_blocks']} blocks, "
              f"{r['recovery_attempts']} recoveries, "
              f"{r.get('strategy_switches', 0)} switches, "
              f"{r.get('rollbacks', 0)} rollbacks")

    # Case 2: Tasks where ours succeeds, all baselines fail
    for r in ours:
        if not r.get("success"):
            continue
        tid = r["task_id"]
        all_fail = all(not baselines[m][tid].get("success", False)
                       for m in baselines if tid in baselines[m])
        if all_fail and r.get("difficulty") in ("medium", "hard"):
            print(f"\n% Cross-method win: {tid} ({r['num_blocks']} blocks, {r['difficulty']})")
            for m in baselines:
                if tid in baselines[m]:
                    br = baselines[m][tid]
                    print(f"    {m}: success={br.get('success')}, steps={br.get('total_steps', 0)}")

    # Case 3: Interesting failures (ours fails despite recovery)
    hard_failures = [r for r in ours
                     if not r.get("success") and r.get("recovery_attempts", 0) > 0
                     and r.get("difficulty") == "hard"]
    print(f"\n% Hard failures with recovery attempts: {len(hard_failures)}")
    for r in hard_failures[:3]:
        print(f"  {r['task_id']}: {r['num_blocks']} blocks, "
              f"reason={r.get('failure_reason', 'unknown')[:60]}, "
              f"recoveries={r.get('recovery_attempts', 0)}")


### Main ###

if __name__ == "__main__":
    if os.path.exists(os.path.join(RESULTS_DIR, "main_comparison.json")):
        print("=" * 80)
        print("=== Main Comparison ===")
        print("=" * 80)
        results = load_results("main_comparison.json")
        print_main_table(results)
        print_efficiency_table(results)
        print_optimality_analysis(results)
        print_significance_tests(results)
        print_recovery_analysis(results)
        analyze_failure_modes(results)
        print_cross_method_analysis(results)
        find_interesting_cases(results)

    if os.path.exists(os.path.join(RESULTS_DIR, "ablation.json")):
        print("\n" + "=" * 80)
        print("=== Ablation Study ===")
        print("=" * 80)
        results = load_results("ablation.json")
        print_ablation_table(results)
        print_recovery_analysis(results)

    if os.path.exists(os.path.join(RESULTS_DIR, "scaling.json")):
        print("\n" + "=" * 80)
        print("=== Scaling Experiment ===")
        print("=" * 80)
        results = load_results("scaling.json")
        for method_name, method_results in results.items():
            m = compute_metrics(method_results)
            by_blocks = {}
            for r in method_results:
                nb = r["num_blocks"]
                if nb not in by_blocks:
                    by_blocks[nb] = []
                by_blocks[nb].append(r.get("success", False))
            print(f"\n{method_name}:")
            for nb in sorted(by_blocks):
                rate = sum(by_blocks[nb]) / len(by_blocks[nb]) * 100
                print(f"  {nb} blocks: {rate:.0f}% ({sum(by_blocks[nb])}/{len(by_blocks[nb])})")

    if os.path.exists(os.path.join(RESULTS_DIR, "model_comparison.json")):
        print("\n" + "=" * 80)
        print("=== Model Comparison (GPT-4o-mini) ===")
        print("=" * 80)
        results = load_results("model_comparison.json")
        for method_name, method_results in results.items():
            m = compute_metrics(method_results)
            print(f"{method_name}: {m['overall_rate']:.0f}% overall "
                  f"(easy={m.get('easy_rate', 0):.0f}%, "
                  f"medium={m.get('medium_rate', 0):.0f}%, "
                  f"hard={m.get('hard_rate', 0):.0f}%)")
