# Failure handler - tries to recover when plans break

import threading
from .decomposer import redecompose_task, extract_actions
from .executor import execute_subtask, get_alternative_strategy
from .verifier import VerificationResult, analyze_failure
from .llm import call_llm, parse_json


class RecoveryAction:
    STRATEGY_SWITCH = "strategy_switch"
    REPAIR = "repair"
    ROLLBACK = "rollback"
    PROPAGATE = "propagate"


class RecoveryResult:
    def __init__(self, action_taken: str, new_actions: list = None, success: bool = False):
        self.action_taken = action_taken
        self.new_actions = new_actions or []
        self.success = success

    def __repr__(self):
        return f"RecoveryResult(action={self.action_taken}, success={self.success}, n_actions={len(self.new_actions)})"


# Thread-local recovery state
_local = threading.local()


def reset_recovery_state():
    """Reset for a new task."""
    _local.tried_tiers = set()


def repair_plan(actions, failed_step, current_state_text, goal_state_text, diagnostic=None):
    """Try to fix the plan from the failure point without redoing everything."""
    completed = actions[:failed_step]
    failed_action = actions[failed_step] if failed_step < len(actions) else "unknown"

    diag_text = ""
    if diagnostic:
        parts = []
        if diagnostic.blocking_chain:
            parts.append("Blocking chain: " + " -> ".join(diagnostic.blocking_chain))
        if diagnostic.suggested_fix:
            parts.append("Fix hint: " + diagnostic.suggested_fix)
        if parts:
            diag_text = "\n" + "\n".join(parts)

    prompt = f"""A BlocksWorld plan partially succeeded. Generate the remaining actions.

Steps completed successfully: {completed if completed else '(none)'}
Current state (after successful steps): {current_state_text}
Goal state: {goal_state_text}

Step that failed: "{failed_action}"
{diag_text}

Generate ONLY the remaining actions needed from the current state to reach the goal.
Available actions: pick-up X, put-down X, stack X Y, unstack X Y.
Return JSON: {{"actions": ["action1", "action2", ...]}}"""

    system = """You are a BlocksWorld action planner. Generate precise action sequences.
Rules: Can only hold one block. Hand must be empty to pick-up/unstack. Block must be clear to manipulate."""

    try:
        response = call_llm(prompt, system=system, json_mode=True, temperature=0.2)
        result = parse_json(response)
        return result.get("actions", [])
    except Exception:
        return []


def handle_failure(verification: VerificationResult, subtask: dict,
                   current_state_text: str, goal_state_text: str,
                   current_strategy: str = None, attempt: int = 0,
                   max_attempts: int = 3, actions=None,
                   current_state=None, goal_state=None) -> RecoveryResult:
    """Try to recover from a failure. Goes through 4 tiers:
    switch strategy -> surgical repair -> full rollback -> give up."""
    tried_tiers = getattr(_local, "tried_tiers", set())

    if attempt >= max_attempts:
        return RecoveryResult(RecoveryAction.PROPAGATE, success=False)

    # Compute diagnostic for all tiers to use
    diagnostic = analyze_failure(
        verification.failed_action, verification.error,
        verification.state_at_failure
    ) if verification.state_at_failure else None

    subtask_type = subtask.get("type", "procedural")

    # Tier 1: Strategy switch (try if not already tried)
    if "strategy_switch" not in tried_tiers and current_strategy:
        alt_strategy = get_alternative_strategy(subtask_type, current_strategy)
        if alt_strategy != current_strategy:
            tried_tiers.add("strategy_switch")
            _local.tried_tiers = tried_tiers
            new_actions = execute_subtask(subtask, current_state_text, strategy=alt_strategy)
            if new_actions:
                return RecoveryResult(
                    RecoveryAction.STRATEGY_SWITCH,
                    new_actions=new_actions,
                    success=True,
                )

    # Tier 2: Surgical repair (fix from failure point)
    if "repair" not in tried_tiers and actions and verification.step_index is not None:
        tried_tiers.add("repair")
        _local.tried_tiers = tried_tiers
        repair_actions = repair_plan(
            actions, verification.step_index,
            current_state_text, goal_state_text,
            diagnostic=diagnostic,
        )
        if repair_actions:
            return RecoveryResult(
                RecoveryAction.REPAIR,
                new_actions=repair_actions,
                success=True,
            )

    # Tier 3: Rollback (re-decompose from current state with diagnostics)
    if "rollback" not in tried_tiers:
        tried_tiers.add("rollback")
        _local.tried_tiers = tried_tiers
        failed_action = verification.failed_action or "unknown"
        error_msg = verification.error or "unknown error"
        try:
            new_tree = redecompose_task(
                current_state_text, goal_state_text,
                failed_action, error_msg,
                diagnostic=diagnostic,
                current_state=current_state,
                goal_state=goal_state,
            )
            new_actions = extract_actions(new_tree)
            if new_actions:
                return RecoveryResult(
                    RecoveryAction.ROLLBACK,
                    new_actions=new_actions,
                    success=True,
                )
        except Exception:
            pass

    # Tier 4: Propagate failure
    return RecoveryResult(RecoveryAction.PROPAGATE, success=False)
