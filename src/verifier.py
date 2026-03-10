# Verifier - simulates plans and catches failures

from dataclasses import dataclass, field
from .blocksworld import BlocksWorldEnv, BlocksWorldState


@dataclass
class FailureDiagnostic:
    error_type: str = "unknown"  # blocking_dependency, position_wrong, block_missing, holding_conflict, goal_incomplete
    blocking_chain: list = field(default_factory=list)
    suggested_fix: str = ""
    expected: str = ""
    actual: str = ""


def analyze_failure(action: str, error: str, state: BlocksWorldState) -> FailureDiagnostic:
    """Figure out why an action failed and give useful diagnostic info."""
    diag = FailureDiagnostic()
    if not action or not error:
        diag.error_type = "goal_incomplete"
        diag.suggested_fix = "Plan executed but did not reach goal. Check which goal conditions are unmet."
        return diag

    parts = action.strip().split()
    act = parts[0].lower() if parts else ""
    blocks_in_action = [p.upper() for p in parts[1:]]

    error_lower = error.lower()

    if "does not exist" in error_lower or "doesn't exist" in error_lower:
        diag.error_type = "block_missing"
        bad_block = blocks_in_action[0] if blocks_in_action else "?"
        diag.suggested_fix = f"Block {bad_block} does not exist in this task. Check block names: {sorted(state.on.keys())}."
        return diag

    if "not clear" in error_lower:
        diag.error_type = "blocking_dependency"
        blocked = blocks_in_action[0] if blocks_in_action else None
        if blocked and blocked in state.on:
            chain = [blocked]
            current = blocked
            while True:
                on_top = state.what_is_on(current)
                if on_top is None:
                    break
                chain.append(on_top)
                current = on_top
            chain.reverse()
            diag.blocking_chain = [f"{b} is on {state.on.get(b, '?')}" for b in chain]
            top = chain[0]
            diag.suggested_fix = f"First unstack {top} (the top of the blocking stack), then work down to free {blocked}."
        return diag

    if "not on" in error_lower:
        diag.error_type = "position_wrong"
        if len(blocks_in_action) >= 2:
            block, expected_support = blocks_in_action[0], blocks_in_action[1]
            actual_loc = state.on.get(block, "unknown")
            diag.expected = f"{block} on {expected_support}"
            diag.actual = f"{block} on {actual_loc}"
            diag.suggested_fix = f"{block} is actually on {actual_loc}, not on {expected_support}. Adjust the plan accordingly."
        return diag

    if "not on table" in error_lower:
        diag.error_type = "position_wrong"
        block = blocks_in_action[0] if blocks_in_action else "?"
        actual_loc = state.on.get(block, "unknown")
        diag.suggested_fix = f"{block} is on {actual_loc}, not on the table. Unstack it first."
        return diag

    if "holding" in error_lower or "hand" in error_lower:
        diag.error_type = "holding_conflict"
        diag.suggested_fix = "Hand state mismatch. Put down the currently held block first, or pick up before stacking."
        return diag

    diag.error_type = "unknown"
    diag.suggested_fix = f"Action '{action}' failed: {error}"
    return diag


class VerificationResult:
    def __init__(self, valid: bool, failed_action: str = None, error: str = None,
                 step_index: int = None, state_at_failure: BlocksWorldState = None):
        self.valid = valid
        self.failed_action = failed_action
        self.error = error
        self.step_index = step_index
        self.state_at_failure = state_at_failure

    def __repr__(self):
        if self.valid:
            return "VerificationResult(valid=True)"
        return f"VerificationResult(valid=False, step={self.step_index}, action='{self.failed_action}', error='{self.error}')"


def verify_plan(actions: list, initial_state: BlocksWorldState, goal_state: BlocksWorldState) -> VerificationResult:
    """Simulate the plan step by step, return where it breaks (if it does)."""
    env = BlocksWorldEnv(initial_state, goal_state)
    env.reset()

    for i, action in enumerate(actions):
        success, msg = env.execute_action(action)
        if not success:
            return VerificationResult(
                valid=False,
                failed_action=action,
                error=msg,
                step_index=i,
                state_at_failure=env.state.copy(),
            )

    if env.is_goal_reached():
        return VerificationResult(valid=True)
    else:
        return VerificationResult(
            valid=False,
            failed_action=None,
            error="Plan completed but goal not reached.",
            step_index=len(actions),
            state_at_failure=env.state.copy(),
        )


def verify_action_preconditions(action: str, state: BlocksWorldState, holding: str = None) -> tuple:
    """Check preconditions without executing. Returns (valid, error_msg)."""
    parts = action.strip().split()
    if len(parts) < 2:
        return False, f"Invalid action format: {action}"

    act = parts[0].lower()
    if act == "pick-up":
        block = parts[1].upper()
        if holding is not None:
            return False, f"Hand not empty (holding {holding})"
        if block not in state.on:
            return False, f"Block {block} doesn't exist"
        if state.on[block] != "table":
            return False, f"{block} not on table"
        if not state.is_clear(block):
            return False, f"{block} not clear"
        return True, None

    elif act == "put-down":
        block = parts[1].upper()
        if holding != block:
            return False, f"Not holding {block}"
        return True, None

    elif act == "stack":
        if len(parts) < 3:
            return False, "Stack needs two args"
        block, target = parts[1].upper(), parts[2].upper()
        if holding != block:
            return False, f"Not holding {block}"
        if target not in state.on:
            return False, f"{target} doesn't exist"
        if not state.is_clear(target):
            return False, f"{target} not clear"
        return True, None

    elif act == "unstack":
        if len(parts) < 3:
            return False, "Unstack needs two args"
        block, source = parts[1].upper(), parts[2].upper()
        if holding is not None:
            return False, f"Hand not empty (holding {holding})"
        if block not in state.on:
            return False, f"{block} doesn't exist"
        if state.on.get(block) != source:
            return False, f"{block} not on {source}"
        if not state.is_clear(block):
            return False, f"{block} not clear"
        return True, None

    return False, f"Unknown action: {act}"
