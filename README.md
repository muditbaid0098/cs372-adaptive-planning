# Adaptive Hierarchical Planning for BlocksWorld

CS 372 project. Uses GPT-4o to solve BlocksWorld tasks through a 5-module pipeline:

1. **Decomposer** - breaks the task into a hierarchical subtask tree
2. **Classifier** - tags subtasks by reasoning type (spatial, procedural, logical, etc.)
3. **Strategy Executor** - picks a reasoning strategy (CoT, ToT, precondition-checking, state-tracking) based on subtask type
4. **Verifier** - simulates the plan to catch failures before execution
5. **Failure Handler** - 4-tier recovery: strategy switch -> surgical repair -> full re-plan -> give up

The key idea is that different subtasks need different reasoning strategies, and when things go wrong, the system can try alternative approaches instead of just failing.

## Setup

```bash
pip install openai python-dotenv numpy scipy
```

Create a `.env` file in the project root with your OpenAI API key:
```
OPENAI_API_KEY=sk-...
```

## Running experiments

Run everything (takes a while and uses a lot of API calls):
```bash
python run_experiments.py --experiment main      # ours vs baselines
python run_experiments.py --experiment ablation   # ablation study
python run_experiments.py --experiment scaling    # 9-10 block tasks
python run_experiments.py --experiment all        # all of the above
```

Use `--small` for a quick test with fewer tasks (5 per difficulty instead of 20):
```bash
python run_experiments.py --experiment main --small
```

There are also standalone scripts if you want to re-run specific parts:
```bash
python run_ours_only.py          # just our method (keeps baseline results)
python run_baselines_only.py     # just the baselines
python run_ablation_only.py      # just the ablation study
```

## Analyzing results

```bash
python analyze_results.py
```

This prints LaTeX tables, significance tests, recovery analysis, etc. Results are read from `results/*.json`.

## Project structure

```
src/
  blocksworld.py    - environment simulator, task generation, BFS optimal solver
  llm.py            - OpenAI API wrapper with retries
  decomposer.py     - task decomposition (module 1)
  classifier.py     - subtask classification (module 2)
  executor.py       - strategy execution (module 3)
  verifier.py       - plan verification (module 4)
  failure_handler.py - recovery logic (module 5)
  planner.py        - orchestrates everything
  baselines.py      - flat CoT, flat ToT, ReAct implementations

run_experiments.py      - main experiment runner
run_ours_only.py        - run just our method
run_baselines_only.py   - run just baselines
run_ablation_only.py    - run ablation study
analyze_results.py      - generate tables and analysis

results/                - experiment results (JSON)
```

## Results summary

On 60 BlocksWorld tasks (20 easy, 20 medium, 20 hard):

| Method | Easy | Medium | Hard | Overall |
|--------|------|--------|------|---------|
| Flat CoT | 95% | 65% | 50% | 70% |
| Flat ToT | 85% | 40% | 20% | 48% |
| ReAct | 85% | 65% | 20% | 57% |
| **Ours** | **95%** | **85%** | **65%** | **82%** |

The biggest gains are on medium and hard tasks where recovery and strategy switching actually matter.
