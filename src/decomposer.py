# Decomposer - breaks tasks into subtask trees

from .llm import call_llm, parse_json
from .blocksworld import BlocksWorldState

DECOMPOSE_SYSTEM = """You are a hierarchical task planner for BlocksWorld.
BlocksWorld has blocks labeled with letters. Blocks can be on the table or stacked on other blocks.
Available actions and their preconditions:
- pick-up X: hand must be empty, X must be on the table, X must be clear (nothing on top)
- put-down X: must be holding X
- stack X Y: must be holding X, Y must be clear (nothing on top)
- unstack X Y: hand must be empty, X must be on Y, X must be clear (nothing on top)

Critical rules:
- Can only hold ONE block at a time.
- Must put down / stack a held block before picking up / unstacking another.
- To move a block that has other blocks on top, you must first move ALL blocks above it.
- Order matters: work top-down when clearing stacks, bottom-up when building them."""

DECOMPOSE_PROMPT = """Decompose this BlocksWorld task into a hierarchical plan.

Current state: {current_state}
Goal state: {goal_state}
{state_diff}
Create a JSON task tree. Each node has:
- "goal": description of what this subtask achieves
- "type": one of "spatial", "procedural", "logical", "arithmetic", "commonsense"
- "children": list of subtasks (empty for leaf/action nodes)
- "actions": list of BlocksWorld actions for leaf nodes only (e.g., ["unstack C B", "put-down C"])

Rules:
- The root should describe the overall goal
- Mid-level nodes group related moves (e.g., "clear block B", "build stack A-B-C")
- Leaf nodes contain the actual actions
- Max depth: 3 levels
- Actions must be valid: pick-up X (from table), put-down X (to table), stack X Y (put X on Y), unstack X Y (remove X from Y)
- You must hold a block before stacking/putting down, and your hand must be empty before picking up/unstacking

Think carefully about the correct order of operations. To move block X that has Y on top, you must first move Y.

Return ONLY valid JSON."""


def compute_state_diff(current_state, goal_state):
    """Diff between current and goal states. Returns empty string for text inputs."""
    if isinstance(current_state, str) or isinstance(goal_state, str):
        return ""
    cur_pos = current_state.get_positions()
    goal_pos = goal_state.get_positions()
    moves = []
    correct = []
    for block in sorted(set(cur_pos) | set(goal_pos)):
        cur = cur_pos.get(block, "missing")
        goal = goal_pos.get(block, "missing")
        if cur != goal:
            moves.append(f"  - {block}: currently {cur} -> goal: {goal}")
        else:
            correct.append(block)
    if not moves:
        return "\nAll blocks are already in the correct position."
    lines = ["\nBlocks that need to move:"]
    lines.extend(moves)
    if correct:
        lines.append(f"Blocks already correct: [{', '.join(correct)}]")
    return "\n".join(lines)


def decompose_task(current_state_text: str, goal_state_text: str,
                   current_state=None, goal_state=None) -> dict:
    """Call the LLM to decompose a task into a subtask tree."""
    diff = compute_state_diff(current_state, goal_state) if current_state and goal_state else ""
    prompt = DECOMPOSE_PROMPT.format(
        current_state=current_state_text,
        goal_state=goal_state_text,
        state_diff=diff,
    )
    response = call_llm(prompt, system=DECOMPOSE_SYSTEM, json_mode=True, temperature=0.2)
    tree = parse_json(response)
    return tree


def extract_actions(tree: dict) -> list:
    """DFS to pull out all actions from the tree."""
    actions = []
    if tree.get("children"):
        for child in tree["children"]:
            actions.extend(extract_actions(child))
    elif tree.get("actions"):
        actions.extend(tree["actions"])
    return actions


def redecompose_task(current_state_text: str, goal_state_text: str,
                     failed_action: str, error_msg: str,
                     diagnostic=None, current_state=None, goal_state=None) -> dict:
    """Re-plan after a failure, starting from the current (partially executed) state."""
    diff = compute_state_diff(current_state, goal_state) if current_state and goal_state else ""

    diag_text = ""
    if diagnostic:
        diag_parts = []
        if diagnostic.blocking_chain:
            diag_parts.append("Blocking chain: " + " -> ".join(diagnostic.blocking_chain))
        if diagnostic.suggested_fix:
            diag_parts.append("Suggested approach: " + diagnostic.suggested_fix)
        if diag_parts:
            diag_text = "\n\nFailure analysis:\n" + "\n".join(diag_parts)

    prompt = f"""The previous plan failed. Re-plan from the current state.

Current state: {current_state_text}
Goal state: {goal_state_text}
{diff}
The action "{failed_action}" failed with error: {error_msg}{diag_text}

Create a new hierarchical plan from the current state to reach the goal.
Use the same JSON format with goal, type, children, and actions fields.
Return ONLY valid JSON."""

    response = call_llm(prompt, system=DECOMPOSE_SYSTEM, json_mode=True, temperature=0.3)
    return parse_json(response)
