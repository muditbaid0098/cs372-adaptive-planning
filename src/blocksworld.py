# BlocksWorld environment + task generation

import random
import copy


class BlocksWorldState:
    """State = dict mapping each block -> what it's on (block name or 'table')."""

    def __init__(self, on_relations: dict):
        self.on = dict(on_relations)  # block -> what it's on

    def copy(self):
        return BlocksWorldState(copy.deepcopy(self.on))

    def get_blocks(self):
        return sorted(self.on.keys())

    def is_clear(self, block):
        """Nothing on top of it?"""
        for b, support in self.on.items():
            if support == block:
                return False
        return True

    def is_on_table(self, block):
        return self.on[block] == "table"

    def get_stack(self, block):
        """Get the stack from table up to this block (bottom to top)."""
        stack = [block]
        current = block
        while self.on[current] != "table":
            current = self.on[current]
            stack.append(current)
        stack.reverse()
        return stack

    def get_stacks(self):
        """All stacks, bottom to top."""
        # Find bottom blocks (on table)
        bottoms = [b for b in self.on if self.on[b] == "table"]
        stacks = []
        for bottom in sorted(bottoms):
            stack = [bottom]
            current = bottom
            while True:
                next_block = None
                for b, support in self.on.items():
                    if support == current:
                        next_block = b
                        break
                if next_block is None:
                    break
                stack.append(next_block)
                current = next_block
            stacks.append(stack)
        return stacks

    def matches(self, other):
        """Check if states match."""
        return self.on == other.on

    def to_text(self):
        """Readable text version."""
        stacks = self.get_stacks()
        if not stacks:
            return "Empty table."
        lines = []
        for i, stack in enumerate(stacks):
            if len(stack) == 1:
                lines.append(f"{stack[0]} is on the table")
            else:
                for j in range(len(stack)):
                    if j == 0:
                        lines.append(f"{stack[j]} is on the table")
                    else:
                        lines.append(f"{stack[j]} is on {stack[j-1]}")
        return ". ".join(lines) + "."

    def __repr__(self):
        return f"BlocksWorldState({self.on})"

    def __eq__(self, other):
        return isinstance(other, BlocksWorldState) and self.on == other.on

    def get_positions(self):
        """Dict of block -> 'on X' or 'on table'."""
        return {b: ("on table" if loc == "table" else f"on {loc}")
                for b, loc in self.on.items()}

    def what_is_on(self, block):
        """What's on top of this block? None if clear."""
        for b, support in self.on.items():
            if support == block:
                return b
        return None


class BlocksWorldEnv:
    """Simulates the BlocksWorld -- execute actions, check goal, etc."""

    ACTIONS = ["pick-up", "put-down", "stack", "unstack"]

    def __init__(self, initial_state: BlocksWorldState, goal_state: BlocksWorldState):
        self.initial_state = initial_state.copy()
        self.goal_state = goal_state
        self.state = initial_state.copy()
        self.holding = None
        self.steps = 0
        self.max_steps = 50
        self.action_history = []

    def reset(self):
        self.state = self.initial_state.copy()
        self.holding = None
        self.steps = 0
        self.action_history = []
        return self.get_observation()

    def get_observation(self):
        obs = f"Current state: {self.state.to_text()}\n"
        obs += f"Goal: {self.goal_state.to_text()}\n"
        if self.holding:
            obs += f"Holding: {self.holding}\n"
        else:
            obs += "Hand is empty.\n"
        return obs

    def execute_action(self, action_str: str):
        """Execute an action like 'pick-up A'. Returns (success, message)."""
        self.steps += 1
        if self.steps > self.max_steps:
            return False, "Max steps exceeded."

        parts = action_str.strip().split()
        if len(parts) < 2:
            return False, f"Invalid action format: {action_str}"

        action = parts[0].lower()
        self.action_history.append(action_str)

        if action == "pick-up":
            block = parts[1].upper()
            return self._pick_up(block)
        elif action == "put-down":
            block = parts[1].upper()
            return self._put_down(block)
        elif action == "stack":
            if len(parts) < 3:
                return False, f"Stack requires two blocks: {action_str}"
            block = parts[1].upper()
            target = parts[2].upper()
            return self._stack(block, target)
        elif action == "unstack":
            if len(parts) < 3:
                return False, f"Unstack requires two blocks: {action_str}"
            block = parts[1].upper()
            source = parts[2].upper()
            return self._unstack(block, source)
        else:
            return False, f"Unknown action: {action}"

    def _pick_up(self, block):
        if self.holding is not None:
            return False, f"Cannot pick up {block}: already holding {self.holding}."
        if block not in self.state.on:
            return False, f"Block {block} does not exist."
        if not self.state.is_on_table(block):
            return False, f"Cannot pick up {block}: it's not on the table (use unstack)."
        if not self.state.is_clear(block):
            return False, f"Cannot pick up {block}: it's not clear."
        self.holding = block
        del self.state.on[block]
        return True, f"Picked up {block}."

    def _put_down(self, block):
        if self.holding != block:
            return False, f"Cannot put down {block}: not holding it."
        self.state.on[block] = "table"
        self.holding = None
        return True, f"Put down {block} on table."

    def _stack(self, block, target):
        if self.holding != block:
            return False, f"Cannot stack {block}: not holding it."
        if target not in self.state.on:
            return False, f"Block {target} does not exist."
        if not self.state.is_clear(target):
            return False, f"Cannot stack on {target}: it's not clear."
        self.state.on[block] = target
        self.holding = None
        return True, f"Stacked {block} on {target}."

    def _unstack(self, block, source):
        if self.holding is not None:
            return False, f"Cannot unstack {block}: already holding {self.holding}."
        if block not in self.state.on:
            return False, f"Block {block} does not exist."
        if self.state.on.get(block) != source:
            return False, f"Cannot unstack {block} from {source}: {block} is not on {source}."
        if not self.state.is_clear(block):
            return False, f"Cannot unstack {block}: it's not clear."
        self.holding = block
        del self.state.on[block]
        return True, f"Unstacked {block} from {source}."

    def is_goal_reached(self):
        if self.holding is not None:
            return False
        # Rebuild state with holding block removed
        return self.state.on == self.goal_state.on

    def get_status(self):
        if self.is_goal_reached():
            return "success"
        if self.steps >= self.max_steps:
            return "max_steps"
        return "in_progress"


def bfs_optimal_plan(initial: BlocksWorldState, goal: BlocksWorldState, max_states: int = 500000) -> tuple:
    """BFS to find optimal plan. Returns (length, actions). Gives up after max_states."""
    from collections import deque

    blocks = sorted(initial.on.keys())
    block_to_idx = {b: i for i, b in enumerate(blocks)}
    n = len(blocks)

    def encode_state(state_on, holding):
        # encode as tuple for hashing
        arr = [0] * n
        for b, support in state_on.items():
            idx = block_to_idx[b]
            if support == "table":
                arr[idx] = 0
            else:
                arr[idx] = block_to_idx[support] + 1
        h = block_to_idx[holding] if holding else -1
        return (tuple(arr), h)

    def get_valid_actions(state_on, holding):
        actions = []
        supports = set(state_on.values()) - {"table"}
        if holding is None:
            for b in blocks:
                if b in state_on and b not in supports:
                    if state_on[b] == "table":
                        actions.append(("pick-up", b, None))
                    else:
                        actions.append(("unstack", b, state_on[b]))
        else:
            actions.append(("put-down", holding, None))
            for b in blocks:
                if b in state_on and b not in supports:
                    actions.append(("stack", holding, b))
        return actions

    def apply_action(state_on, holding, action):
        new_on = dict(state_on)
        act, block, target = action
        if act == "pick-up":
            del new_on[block]
            return new_on, block
        elif act == "put-down":
            new_on[block] = "table"
            return new_on, None
        elif act == "stack":
            new_on[block] = target
            return new_on, None
        elif act == "unstack":
            del new_on[block]
            return new_on, block
        return new_on, holding

    def action_str(action):
        act, block, target = action
        return f"{act} {block} {target}" if target else f"{act} {block}"

    goal_on = goal.on
    start_key = encode_state(initial.on, None)
    visited = {start_key}
    queue = deque()
    queue.append((dict(initial.on), None, []))

    while queue and len(visited) < max_states:
        current_on, holding, path = queue.popleft()

        if holding is None and current_on == goal_on:
            return len(path), [action_str(a) for a in path]

        for action in get_valid_actions(current_on, holding):
            new_on, new_holding = apply_action(current_on, holding, action)
            key = encode_state(new_on, new_holding)
            if key not in visited:
                visited.add(key)
                queue.append((new_on, new_holding, path + [action]))

    return -1, []  # No solution found or state limit reached


def generate_task(num_blocks: int, seed: int = None):
    """Generate a random task with n blocks."""
    if seed is not None:
        random.seed(seed)

    blocks = [chr(ord("A") + i) for i in range(num_blocks)]

    def random_state(blocks):
        blocks = list(blocks)
        random.shuffle(blocks)
        on = {}
        remaining = list(blocks)
        while remaining:
            max_stack = min(len(remaining), len(blocks))
            stack_size = random.randint(1, max_stack)
            stack = remaining[:stack_size]
            remaining = remaining[stack_size:]
            on[stack[0]] = "table"
            for i in range(1, len(stack)):
                on[stack[i]] = stack[i - 1]
        return BlocksWorldState(on)

    initial = random_state(blocks)
    goal = random_state(blocks)

    # Make sure they're different
    while initial.on == goal.on:
        goal = random_state(blocks)

    return initial, goal


def generate_benchmark(easy=20, medium=20, hard=20, seed=42):
    """Generate the benchmark: easy (3-4 blocks), medium (5-6), hard (7-8)."""
    tasks = []
    random.seed(seed)

    for i in range(easy):
        num_blocks = random.choice([3, 4])
        init, goal = generate_task(num_blocks, seed=seed + i)
        tasks.append({"id": f"easy_{i}", "difficulty": "easy", "num_blocks": num_blocks,
                       "initial": init, "goal": goal})

    for i in range(medium):
        num_blocks = random.choice([5, 6])
        init, goal = generate_task(num_blocks, seed=seed + 100 + i)
        tasks.append({"id": f"medium_{i}", "difficulty": "medium", "num_blocks": num_blocks,
                       "initial": init, "goal": goal})

    for i in range(hard):
        num_blocks = random.choice([7, 8])
        init, goal = generate_task(num_blocks, seed=seed + 200 + i)
        tasks.append({"id": f"hard_{i}", "difficulty": "hard", "num_blocks": num_blocks,
                       "initial": init, "goal": goal})

    return tasks
