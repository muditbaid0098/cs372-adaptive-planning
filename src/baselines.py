# Baselines: flat CoT, flat ToT, ReAct

from .llm import call_llm, parse_json
from .blocksworld import BlocksWorldEnv, BlocksWorldState
from .verifier import verify_plan, VerificationResult

BLOCKSWORLD_SYSTEM = """You are a BlocksWorld planner.
Available actions: pick-up X (from table), put-down X (to table), stack X Y (put X on Y), unstack X Y (remove X from Y).
Rules: Can only hold one block at a time. Must have empty hand to pick-up/unstack. Block must be clear."""


def flat_cot(initial_state: BlocksWorldState, goal_state: BlocksWorldState) -> dict:
    """Flat CoT baseline."""
    prompt = f"""Solve this BlocksWorld task step by step.

Current state: {initial_state.to_text()}
Goal state: {goal_state.to_text()}

Think through the problem step by step, then provide the action sequence.
Return JSON: {{"reasoning": "...", "actions": ["action1", "action2", ...]}}"""

    response = call_llm(prompt, system=BLOCKSWORLD_SYSTEM, json_mode=True, temperature=0.3)
    result = parse_json(response)
    actions = result.get("actions", [])
    verification = verify_plan(actions, initial_state, goal_state)

    return {
        "method": "flat_cot",
        "success": verification.valid,
        "actions": actions,
        "total_steps": len(actions),
    }


def flat_tot(initial_state: BlocksWorldState, goal_state: BlocksWorldState, branches: int = 3) -> dict:
    """Flat ToT baseline."""
    prompt = f"""Solve this BlocksWorld task by exploring {branches} different approaches.

Current state: {initial_state.to_text()}
Goal state: {goal_state.to_text()}

For each approach:
1. Describe the strategy
2. Generate the action sequence
3. Evaluate if it would work

Then select the best approach.

Return JSON: {{
    "approaches": [
        {{"strategy": "...", "actions": [...], "evaluation": "..."}},
        ...
    ],
    "best_index": 0,
    "actions": ["action1", "action2", ...]
}}"""

    response = call_llm(prompt, system=BLOCKSWORLD_SYSTEM, json_mode=True,
                        temperature=0.5, max_tokens=3000)
    result = parse_json(response)
    actions = result.get("actions", [])

    # If no top-level actions, try the best approach
    if not actions:
        approaches = result.get("approaches", [])
        best_idx = result.get("best_index", 0)
        if approaches and 0 <= best_idx < len(approaches):
            actions = approaches[best_idx].get("actions", [])

    verification = verify_plan(actions, initial_state, goal_state)

    return {
        "method": "flat_tot",
        "success": verification.valid,
        "actions": actions,
        "total_steps": len(actions),
    }


def react(initial_state: BlocksWorldState, goal_state: BlocksWorldState,
          max_rounds: int = 15) -> dict:
    """ReAct baseline - think then act in a loop."""
    env = BlocksWorldEnv(initial_state, goal_state)
    env.reset()

    messages = [
        {"role": "system", "content": BLOCKSWORLD_SYSTEM},
        {"role": "user", "content": f"""Solve this BlocksWorld task using interleaved Thought and Action.

Goal state: {goal_state.to_text()}

Current observation:
{env.get_observation()}

At each step, output JSON with:
- "thought": your reasoning about what to do next
- "action": the next action to take (e.g., "unstack C B")
- "done": true if you believe the goal is reached

Provide ONE action at a time."""},
    ]

    all_actions = []
    from .llm import call_llm_multi

    for round_num in range(max_rounds):
        response = call_llm_multi(messages, json_mode=True, temperature=0.3)
        try:
            result = parse_json(response)
        except Exception:
            break

        if result.get("done", False):
            break

        action = result.get("action", "")
        if not action:
            break

        all_actions.append(action)
        success, msg = env.execute_action(action)

        # Add assistant response and observation
        messages.append({"role": "assistant", "content": response})
        obs = env.get_observation()
        feedback = f"Result: {msg}\n{obs}"
        if env.is_goal_reached():
            feedback += "\nGoal reached!"
        messages.append({"role": "user", "content": feedback})

        if env.is_goal_reached():
            break

        if not success:
            messages[-1] = {"role": "user", "content": f"Action failed: {msg}\n{obs}\nTry a different action."}

    return {
        "method": "react",
        "success": env.is_goal_reached(),
        "actions": all_actions,
        "total_steps": len(all_actions),
    }
