# Strategy executor - picks CoT/ToT/precondition/state-tracking based on subtask type

from .llm import call_llm, parse_json

EXECUTOR_SYSTEM = """You are a BlocksWorld action planner. You generate precise action sequences.
Available actions: pick-up X (from table), put-down X (to table), stack X Y (put X on Y), unstack X Y (remove X from Y).
Rules: Can only hold one block at a time. Must have empty hand to pick-up/unstack. Block must be clear to pick-up/unstack/stack-onto."""


def execute_subtask(subtask: dict, state_text: str, strategy: str = None) -> list:
    """Run a subtask with the given strategy, returns action list."""
    subtask_type = subtask.get("type", "procedural")
    goal = subtask.get("goal", "")

    if strategy is None:
        strategy = TYPE_TO_STRATEGY.get(subtask_type, "cot")

    if strategy == "cot":
        return _cot_execute(goal, state_text)
    elif strategy == "tot":
        return _tot_execute(goal, state_text)
    elif strategy == "precondition":
        return _precondition_execute(goal, state_text)
    elif strategy == "state_tracking":
        return _state_tracking_execute(goal, state_text)
    else:
        return _cot_execute(goal, state_text)


TYPE_TO_STRATEGY = {
    "spatial": "state_tracking",
    "procedural": "precondition",
    "logical": "tot",
    "arithmetic": "cot",
    "commonsense": "cot",
}

ALTERNATIVE_STRATEGIES = {
    "spatial": ["state_tracking", "precondition", "cot"],
    "procedural": ["precondition", "state_tracking", "cot"],
    "logical": ["tot", "cot", "precondition"],
    "arithmetic": ["cot", "precondition"],
    "commonsense": ["cot", "state_tracking"],
}


def get_alternative_strategy(subtask_type: str, failed_strategy: str) -> str:
    """Next strategy to try when one fails."""
    alternatives = ALTERNATIVE_STRATEGIES.get(subtask_type, ["cot"])
    for s in alternatives:
        if s != failed_strategy:
            return s
    return "cot"


def _cot_execute(goal: str, state_text: str) -> list:
    """CoT strategy."""
    prompt = f"""Goal: {goal}

{state_text}

Think step by step about what actions are needed. Consider preconditions for each action.
Then provide the action sequence.

Return JSON: {{"reasoning": "step by step thinking", "actions": ["action1", "action2", ...]}}"""

    response = call_llm(prompt, system=EXECUTOR_SYSTEM, json_mode=True, temperature=0.2)
    result = parse_json(response)
    return result.get("actions", [])


def _tot_execute(goal: str, state_text: str) -> list:
    """ToT strategy - generate 3 candidates and pick best."""
    prompt = f"""Goal: {goal}

{state_text}

Generate 3 different possible action sequences to achieve this goal.
For each, explain why it might work or fail.
Then select the best one.

Return JSON: {{
    "candidates": [
        {{"actions": [...], "reasoning": "..."}},
        {{"actions": [...], "reasoning": "..."}},
        {{"actions": [...], "reasoning": "..."}}
    ],
    "best_index": 0,
    "selection_reasoning": "..."
}}"""

    response = call_llm(prompt, system=EXECUTOR_SYSTEM, json_mode=True, temperature=0.5, max_tokens=3000)
    result = parse_json(response)
    candidates = result.get("candidates", [])
    best_idx = result.get("best_index", 0)
    if candidates and 0 <= best_idx < len(candidates):
        return candidates[best_idx].get("actions", [])
    elif candidates:
        return candidates[0].get("actions", [])
    return []


def _precondition_execute(goal: str, state_text: str) -> list:
    """Check preconditions explicitly before each action."""
    prompt = f"""Goal: {goal}

{state_text}

For each action, explicitly check its preconditions BEFORE adding it to the plan:
- pick-up X: hand empty, X on table, X is clear
- put-down X: holding X
- stack X Y: holding X, Y is clear
- unstack X Y: hand empty, X on Y, X is clear

List each action with its precondition check. Only include actions whose preconditions are met
given the state after all previous actions.

Return JSON: {{
    "plan": [
        {{"action": "...", "preconditions_met": true, "state_after": "..."}},
        ...
    ],
    "actions": ["action1", "action2", ...]
}}"""

    response = call_llm(prompt, system=EXECUTOR_SYSTEM, json_mode=True, temperature=0.2, max_tokens=3000)
    result = parse_json(response)
    return result.get("actions", [])


def _state_tracking_execute(goal: str, state_text: str) -> list:
    """Track full world state after each action."""
    prompt = f"""Goal: {goal}

{state_text}

Maintain an explicit world model. After each action, write out the complete state.
Track: which blocks are on what, which are clear, what you're holding.

Generate actions one by one, updating the state each time, until the goal is reached.

Return JSON: {{
    "trace": [
        {{"action": "...", "state_after": "...", "holding": "..."}},
        ...
    ],
    "actions": ["action1", "action2", ...]
}}"""

    response = call_llm(prompt, system=EXECUTOR_SYSTEM, json_mode=True, temperature=0.2, max_tokens=3000)
    result = parse_json(response)
    return result.get("actions", [])
