# Classifier - tags subtasks with reasoning types (spatial, procedural, etc.)

from .llm import call_llm, parse_json

CLASSIFY_SYSTEM = """You classify planning subtasks into reasoning types.
Types:
- spatial: involves physical arrangement, positions, locations (e.g., "move block A on top of B")
- procedural: involves following a sequence of steps with preconditions (e.g., "clear block A then move it")
- logical: involves deduction, constraint satisfaction (e.g., "determine which block to move first")
- arithmetic: involves counting or calculation (e.g., "calculate number of moves needed")
- commonsense: involves world knowledge (e.g., "find the right tool")

For BlocksWorld, most subtasks are spatial or procedural."""

CLASSIFY_PROMPT = """Classify each subtask in this task tree by reasoning type.

Task tree:
{tree_json}

For each node, verify or update the "type" field to be one of: spatial, procedural, logical, arithmetic, commonsense.

Consider context: a subtask's parent and siblings help determine its type.
For example, "move block A" under "rearrange stack" is spatial, not procedural.

Return the updated tree as JSON with the same structure but corrected type fields.
Return ONLY valid JSON."""


def classify_tree(tree: dict) -> dict:
    """Classify all nodes in a task tree."""
    import json
    prompt = CLASSIFY_PROMPT.format(tree_json=json.dumps(tree, indent=2))
    response = call_llm(prompt, system=CLASSIFY_SYSTEM, json_mode=True, temperature=0.1)
    classified = parse_json(response)
    return classified


def classify_single(subtask_goal: str, parent_goal: str = None, sibling_goals: list = None) -> str:
    """Classify one subtask (with parent/sibling context if available)."""
    context = f"Subtask: {subtask_goal}"
    if parent_goal:
        context += f"\nParent goal: {parent_goal}"
    if sibling_goals:
        context += f"\nSibling goals: {', '.join(sibling_goals)}"

    prompt = f"""{context}

Classify this subtask as one of: spatial, procedural, logical, arithmetic, commonsense.
Return JSON: {{"type": "<type>", "confidence": <0-1>}}"""

    response = call_llm(prompt, system=CLASSIFY_SYSTEM, json_mode=True, temperature=0.1)
    result = parse_json(response)
    return result.get("type", "procedural")
