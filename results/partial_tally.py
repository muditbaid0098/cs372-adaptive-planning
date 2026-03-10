"""Quick tally of partial results from the running experiment."""

# Parsed from output of the full sequential run (b93qfg05a)
# Easy: 20/20 done, Medium: 20/20 done, Hard: 4/20 done (hard_0 through hard_3)

results = {
    "flat_cot": {
        "easy": {"ok": 18, "fail": 2, "total": 20},   # FAIL: easy_10, (1 more)
        "medium": {"ok": 12, "fail": 8, "total": 20},
        "hard": {"ok": 1, "fail": 3, "total": 4},      # hard_3 OK
    },
    "flat_tot": {
        "easy": {"ok": 16, "fail": 4, "total": 20},
        "medium": {"ok": 9, "fail": 11, "total": 20},
        "hard": {"ok": 1, "fail": 3, "total": 4},      # hard_3 OK
    },
    "react": {
        "easy": {"ok": 19, "fail": 1, "total": 20},
        "medium": {"ok": 13, "fail": 7, "total": 20},
        "hard": {"ok": 1, "fail": 3, "total": 4},      # hard_2 OK
    },
    "ours": {
        "easy": {"ok": 14, "fail": 6, "total": 20},
        "medium": {"ok": 14, "fail": 6, "total": 20},
        "hard": {"ok": 2, "fail": 2, "total": 4},      # hard_1, hard_2 OK
    },
}

print("=" * 70)
print(f"{'Method':<15} {'Easy':>10} {'Medium':>10} {'Hard(4)':>10} {'Overall':>10}")
print("=" * 70)
for method, data in results.items():
    easy_rate = data["easy"]["ok"] / data["easy"]["total"] * 100
    med_rate = data["medium"]["ok"] / data["medium"]["total"] * 100
    hard_rate = data["hard"]["ok"] / data["hard"]["total"] * 100
    total_ok = sum(d["ok"] for d in data.values())
    total_n = sum(d["total"] for d in data.values())
    overall = total_ok / total_n * 100
    print(f"{method:<15} {easy_rate:>9.0f}% {med_rate:>9.0f}% {hard_rate:>9.0f}% {overall:>9.0f}%")
print("=" * 70)
print("\nNote: Hard only has 4/20 tasks. Run still in progress.")
print("\nKey observations:")
print("- Ours LEADS on medium (70% vs 65% react, 60% cot)")
print("- Ours is LOWER on easy (70% vs 90% cot, 95% react)")
print("- Hard too few samples but ours=50% vs baselines=25%")
print("- Easy regression is from forcing strategy executor on all leaves")
print("- Fix already applied: use decomposer actions first, strategy only on recovery")
