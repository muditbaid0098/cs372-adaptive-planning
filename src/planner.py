# Main planner - orchestrates the 5 modules

import time
from .blocksworld import BlocksWorldEnv, BlocksWorldState
from .decomposer import decompose_task, extract_actions
from .classifier import classify_tree
from .executor import execute_subtask, TYPE_TO_STRATEGY
from .verifier import verify_plan, VerificationResult
from .failure_handler import handle_failure, RecoveryAction, reset_recovery_state


class PlannerResult:
    def __init__(self):
        self.success = False
        self.actions_executed = []
        self.actions_start_state = None  # state from which actions_executed is valid
        self.total_steps = 0
        self.rollbacks = 0
        self.strategy_switches = 0
        self.recovery_attempts = 0
        self.llm_calls = 0
        self.elapsed_time = 0
        self.failure_reason = None
        self.decomposition_tree = None
        self.recovery_trace = []  # log of recovery events

    def to_dict(self):
        return {
            "success": self.success,
            "total_steps": self.total_steps,
            "rollbacks": self.rollbacks,
            "strategy_switches": self.strategy_switches,
            "recovery_attempts": self.recovery_attempts,
            "failure_reason": self.failure_reason,
            "recovery_trace": self.recovery_trace,
        }


def run_adaptive_planner(initial_state: BlocksWorldState, goal_state: BlocksWorldState,
                         max_recovery_rounds: int = 3, use_classifier: bool = True,
                         use_adaptive_strategy: bool = True,
                         use_verifier: bool = True, use_failure_handler: bool = True) -> PlannerResult:
    """Run the full adaptive planner. Ablation flags disable modules for experiments."""
    from .llm import get_stats, reset_stats
    reset_stats()
    reset_recovery_state()

    result = PlannerResult()
    start_time = time.time()

    current_state = initial_state.copy()
    goal_text = goal_state.to_text()

    for recovery_round in range(max_recovery_rounds + 1):
        current_text = current_state.to_text()

        # Step 1: Decompose (with state-diff when state objects available)
        try:
            tree = decompose_task(current_text, goal_text,
                                  current_state=current_state, goal_state=goal_state)
        except Exception as e:
            result.failure_reason = f"Decomposition failed: {e}"
            break

        # Step 2: Classify (optional)
        if use_classifier:
            try:
                tree = classify_tree(tree)
            except Exception:
                pass  # Use original type hints from decomposer

        result.decomposition_tree = tree

        # Step 3: Extract and optionally re-execute with adaptive strategies
        if use_adaptive_strategy:
            actions = _execute_tree_adaptive(tree, current_text)
        else:
            actions = extract_actions(tree)

        if not actions:
            result.failure_reason = "No actions generated"
            break

        # Step 4: Verify
        if use_verifier:
            verification = verify_plan(actions, current_state, goal_state)
        else:
            # Without verifier, just try to execute
            verification = _execute_and_check(actions, current_state, goal_state)

        if verification.valid:
            result.success = True
            result.actions_executed = actions
            result.actions_start_state = current_state.copy()
            result.total_steps = len(actions)
            break

        # Step 5: Handle failure
        if use_failure_handler and recovery_round < max_recovery_rounds:
            result.recovery_attempts += 1
            round_start_state = current_state.copy()

            # Get current state at failure point
            if verification.state_at_failure:
                current_state = verification.state_at_failure
            current_text = current_state.to_text()

            # Find the subtask that failed
            failed_subtask = _find_failed_subtask(tree, verification.step_index)
            current_strategy = TYPE_TO_STRATEGY.get(
                failed_subtask.get("type", "procedural"), "cot"
            ) if failed_subtask else "cot"

            recovery = handle_failure(
                verification, failed_subtask or tree,
                current_text, goal_text,
                current_strategy=current_strategy,
                attempt=recovery_round,
                actions=actions,
                current_state=current_state,
                goal_state=goal_state,
            )

            # Log recovery event
            trace_entry = {
                "round": recovery_round,
                "failed_action": verification.failed_action,
                "error": verification.error,
                "step_index": verification.step_index,
                "action_taken": recovery.action_taken,
                "strategy_used": current_strategy,
                "n_new_actions": len(recovery.new_actions),
            }

            if recovery.action_taken == RecoveryAction.STRATEGY_SWITCH:
                result.strategy_switches += 1
            elif recovery.action_taken == RecoveryAction.ROLLBACK:
                result.rollbacks += 1
            # REPAIR is tracked via trace but doesn't have a separate counter

            if recovery.success and recovery.new_actions:
                # Verify the recovery plan
                rev = verify_plan(recovery.new_actions, current_state, goal_state)
                trace_entry["recovery_verified"] = rev.valid
                result.recovery_trace.append(trace_entry)
                if rev.valid:
                    result.success = True
                    result.actions_executed = actions[:verification.step_index] + recovery.new_actions
                    result.actions_start_state = round_start_state
                    result.total_steps = len(result.actions_executed)
                    break
                else:
                    # Update state for next recovery round
                    if rev.state_at_failure:
                        current_state = rev.state_at_failure
                    continue
            elif recovery.action_taken == RecoveryAction.PROPAGATE:
                trace_entry["recovery_verified"] = False
                result.recovery_trace.append(trace_entry)
                result.failure_reason = f"Recovery failed: {verification.error}"
                break
        else:
            result.failure_reason = verification.error
            break

    result.elapsed_time = time.time() - start_time
    result.llm_calls = get_stats()["calls"]
    return result


def _execute_tree_adaptive(tree: dict, state_text: str) -> list:
    """Walk the tree and use type-specific strategies for leaves without actions."""
    actions = []
    if tree.get("children"):
        for child in tree["children"]:
            child_actions = _execute_tree_adaptive(child, state_text)
            actions.extend(child_actions)
    elif tree.get("actions"):
        # Leaf with decomposer-provided actions: use them
        actions.extend(tree["actions"])
    elif tree.get("goal"):
        # Leaf without actions: generate via type-specific strategy
        subtask_type = tree.get("type", "procedural")
        strategy = TYPE_TO_STRATEGY.get(subtask_type, "cot")
        generated = execute_subtask(
            {"goal": tree["goal"], "type": subtask_type, "children": []},
            state_text, strategy=strategy
        )
        actions.extend(generated if generated else [])
    return actions


def _find_failed_subtask(tree: dict, failed_step_index: int) -> dict:
    """Find which subtask the failed step belongs to."""
    counter = [0]

    def _search(node):
        if node.get("actions") and not node.get("children"):
            n_actions = len(node["actions"])
            if counter[0] + n_actions > failed_step_index:
                return node
            counter[0] += n_actions
            return None
        for child in node.get("children", []):
            result = _search(child)
            if result:
                return result
        return None

    return _search(tree) or tree


def _execute_and_check(actions: list, initial_state: BlocksWorldState,
                       goal_state: BlocksWorldState) -> VerificationResult:
    """Just run the actions and check -- used when verifier is disabled."""
    env = BlocksWorldEnv(initial_state, goal_state)
    env.reset()
    for i, action in enumerate(actions):
        success, msg = env.execute_action(action)
        if not success:
            return VerificationResult(
                valid=False, failed_action=action, error=msg,
                step_index=i, state_at_failure=env.state.copy(),
            )
    if env.is_goal_reached():
        return VerificationResult(valid=True)
    return VerificationResult(
        valid=False, error="Goal not reached",
        step_index=len(actions), state_at_failure=env.state.copy(),
    )
