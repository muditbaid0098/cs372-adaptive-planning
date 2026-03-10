"""
Streamlit demo for Adaptive Hierarchical Planning in BlocksWorld.
Run: python -m streamlit run demo.py
"""

import sys
import os
import time
import copy
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

import streamlit as st

from src.blocksworld import BlocksWorldState, BlocksWorldEnv, generate_task, bfs_optimal_plan
from src.planner import run_adaptive_planner, PlannerResult
from src.baselines import flat_cot, react
from src.verifier import verify_plan
from src.llm import reset_stats, get_stats

# ─── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="BlocksWorld Adaptive Planner",
    page_icon="",
    layout="wide",
)

# ─── Constants ─────────────────────────────────────────────────────────────────

BLOCK_COLORS = {
    "A": "#e74c3c",  # red
    "B": "#3498db",  # blue
    "C": "#2ecc71",  # green
    "D": "#e67e22",  # orange
    "E": "#9b59b6",  # purple
    "F": "#f1c40f",  # yellow
    "G": "#e91e8a",  # pink
    "H": "#1abc9c",  # cyan
    "I": "#34495e",  # dark gray
    "J": "#95a5a6",  # light gray
}


def get_block_color(block_name):
    return BLOCK_COLORS.get(block_name.upper(), "#7f8c8d")


def text_color_for_bg(hex_color):
    """Return white or black text depending on background brightness."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return "#1a1a1a" if luminance > 0.55 else "#ffffff"


# ─── Visual block rendering ───────────────────────────────────────────────────

def render_blocks(state: BlocksWorldState, label: str = "", holding: str = None, width: int = 400):
    """Render colored block stacks as HTML/CSS."""
    stacks = state.get_stacks()
    max_height = max((len(s) for s in stacks), default=1)
    block_w = 70
    block_h = 44
    gap = 16
    total_w = len(stacks) * (block_w + gap) - gap
    total_w = max(total_w, 150)
    canvas_h = max_height * block_h + 60  # extra for table + label

    html = f'<div style="text-align:center; margin-bottom:8px; font-weight:600; font-size:15px; color:#e0e0e0;">{label}</div>'

    if holding:
        bg = get_block_color(holding)
        tc = text_color_for_bg(bg)
        html += f'''<div style="display:flex; justify-content:center; margin-bottom:6px; align-items:center; gap:8px;">
            <span style="color:#aaa; font-size:12px;">Holding:</span>
            <div style="width:{block_w}px; height:{block_h}px; background:{bg}; border-radius:6px;
                 display:flex; align-items:center; justify-content:center;
                 font-weight:bold; font-size:18px; color:{tc};
                 border:2px dashed #fff; box-shadow:0 2px 8px rgba(0,0,0,0.3);">{holding}</div>
        </div>'''

    html += f'<div style="display:flex; justify-content:center; align-items:flex-end; gap:{gap}px; min-height:{canvas_h}px;">'

    if not stacks:
        html += '<div style="color:#999; font-style:italic;">Empty table</div>'
    else:
        for stack in stacks:
            col_html = '<div style="display:flex; flex-direction:column-reverse; align-items:center;">'
            for block in stack:
                bg = get_block_color(block)
                tc = text_color_for_bg(bg)
                col_html += f'''<div style="width:{block_w}px; height:{block_h}px; background:{bg};
                    border-radius:6px; display:flex; align-items:center; justify-content:center;
                    font-weight:bold; font-size:18px; color:{tc};
                    border:1px solid rgba(255,255,255,0.15);
                    box-shadow:0 2px 6px rgba(0,0,0,0.25); margin-bottom:2px;
                    transition: all 0.3s ease;">{block}</div>'''
            col_html += '</div>'
            html += col_html

    html += '</div>'
    # Table surface
    html += f'''<div style="width:100%; max-width:{max(total_w + 40, 180)}px; height:6px;
        background:linear-gradient(90deg, #5d4e37, #8b7355, #5d4e37);
        border-radius:3px; margin:4px auto 0 auto;"></div>
        <div style="text-align:center; color:#8b7355; font-size:11px; margin-top:2px; font-weight:500;
        letter-spacing:1px;">TABLE</div>'''

    st.markdown(html, unsafe_allow_html=True)


# ─── Simulation helper ────────────────────────────────────────────────────────

def simulate_step(initial_state: BlocksWorldState, actions: list, step: int):
    """Simulate actions up to `step` (0-indexed, inclusive) and return resulting state + holding.
    Returns (state, holding, error_at) where error_at is the step index that failed, or None."""
    dummy_goal = BlocksWorldState(dict(initial_state.on))
    env = BlocksWorldEnv(initial_state.copy(), dummy_goal)
    env.reset()
    error_at = None
    error_msg = None
    for i, action in enumerate(actions):
        if i > step:
            break
        success, msg = env.execute_action(action)
        if not success:
            error_at = i
            error_msg = msg
            break
    return env.state.copy(), env.holding, error_at, error_msg


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("BlocksWorld Planner")
    st.markdown("---")

    num_blocks = st.slider("Number of blocks", 3, 10, 5)
    seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)

    if st.button("Generate Task", use_container_width=True, type="primary"):
        initial, goal = generate_task(num_blocks, seed=int(seed))
        st.session_state["initial"] = initial
        st.session_state["goal"] = goal
        st.session_state["results"] = {}
        st.session_state["running"] = False

    st.markdown("---")
    st.subheader("Methods")
    run_ours = st.checkbox("Our System", value=True)
    run_cot = st.checkbox("Flat CoT", value=True)
    run_react = st.checkbox("ReAct", value=True)

    methods_selected = []
    if run_ours:
        methods_selected.append("ours")
    if run_cot:
        methods_selected.append("flat_cot")
    if run_react:
        methods_selected.append("react")

    st.markdown("---")
    run_btn = st.button("Run Selected Methods", use_container_width=True, type="primary",
                        disabled="initial" not in st.session_state)

    st.markdown("---")
    st.caption("CS 372 -- Adaptive Hierarchical Planning")

# ─── Main area ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    div[data-testid="stMetric"] { background: rgba(30,30,50,0.4); border-radius: 10px; padding: 12px; }
    div[data-testid="stMetric"] label { font-size: 13px !important; }
</style>
""", unsafe_allow_html=True)

st.title("Adaptive Hierarchical Planning for BlocksWorld")
st.caption("Decomposer  ->  Classifier  ->  Strategy Executor  ->  Verifier  ->  Failure Handler")

if "initial" not in st.session_state:
    st.info("Configure the task in the sidebar and click **Generate Task** to begin.")
    st.stop()

initial: BlocksWorldState = st.session_state["initial"]
goal: BlocksWorldState = st.session_state["goal"]

# ─── State display ─────────────────────────────────────────────────────────────

col1, col3 = st.columns(2)

with col1:
    render_blocks(initial, "Initial State")

with col3:
    render_blocks(goal, "Goal State")

st.markdown("---")

# ─── Run methods ───────────────────────────────────────────────────────────────

METHOD_LABELS = {
    "ours": "Our System",
    "flat_cot": "Flat CoT",
    "react": "ReAct",
}

if run_btn:
    if not methods_selected:
        st.warning("Select at least one method.")
        st.stop()

    results = {}
    progress = st.progress(0, text="Running methods...")

    for i, method in enumerate(methods_selected):
        progress.progress((i) / len(methods_selected), text=f"Running {METHOD_LABELS[method]}...")

        reset_stats()
        t0 = time.time()

        if method == "ours":
            res = run_adaptive_planner(initial, goal)
            stats = get_stats()
            elapsed = res.elapsed_time
            results["ours"] = {
                "success": res.success,
                "actions": res.actions_executed,
                "actions_start_state": res.actions_start_state,
                "total_steps": res.total_steps,
                "llm_calls": stats["calls"],
                "tokens": stats["tokens"],
                "time": elapsed,
                "decomposition_tree": res.decomposition_tree,
                "recovery_trace": res.recovery_trace,
                "recovery_attempts": res.recovery_attempts,
                "strategy_switches": res.strategy_switches,
                "rollbacks": res.rollbacks,
                "failure_reason": res.failure_reason,
            }
        elif method == "flat_cot":
            res = flat_cot(initial, goal)
            stats = get_stats()
            elapsed = time.time() - t0
            results["flat_cot"] = {
                "success": res["success"],
                "actions": res["actions"],
                "total_steps": res["total_steps"],
                "llm_calls": stats["calls"],
                "tokens": stats["tokens"],
                "time": elapsed,
            }
        elif method == "react":
            res = react(initial, goal)
            stats = get_stats()
            elapsed = time.time() - t0
            results["react"] = {
                "success": res["success"],
                "actions": res["actions"],
                "total_steps": res["total_steps"],
                "llm_calls": stats["calls"],
                "tokens": stats["tokens"],
                "time": elapsed,
            }

    progress.progress(1.0, text="Done!")
    time.sleep(0.3)
    progress.empty()
    st.session_state["results"] = results

# ─── Display results ──────────────────────────────────────────────────────────

results = st.session_state.get("results", {})

if results:
    st.subheader("Results")

    method_keys = [m for m in ["ours", "flat_cot", "react"] if m in results]
    cols = st.columns(len(method_keys))

    for col, method in zip(cols, method_keys):
        r = results[method]
        with col:
            st.markdown(f"### {METHOD_LABELS[method]}")

            if r["success"]:
                st.success("SUCCESS")
            else:
                st.error("FAILED")
                if method == "ours" and r.get("failure_reason"):
                    st.caption(f"Reason: {r['failure_reason']}")

            # Metrics row
            m1, m2 = st.columns(2)
            m1.metric("Steps", r["total_steps"])
            m2.metric("LLM Calls", r["llm_calls"])
            m3, m4 = st.columns(2)
            m3.metric("Tokens", f"{r['tokens']:,}")
            m4.metric("Time", f"{r['time']:.1f}s")

            # Actions list
            with st.expander("Actions", expanded=False):
                if r["actions"]:
                    for i, a in enumerate(r["actions"], 1):
                        st.text(f"{i}. {a}")
                else:
                    st.caption("No actions generated.")

            # Ours-specific details
            if method == "ours":
                with st.expander("Decomposition Tree"):
                    if r.get("decomposition_tree"):
                        st.json(r["decomposition_tree"])
                    else:
                        st.caption("No tree available.")

                if r.get("recovery_trace"):
                    with st.expander("Recovery Trace", expanded=True):
                        for entry in r["recovery_trace"]:
                            tier = entry.get("action_taken", "unknown")
                            tier_colors = {
                                "repair": "#f39c12",
                                "strategy_switch": "#e67e22",
                                "rollback": "#e74c3c",
                                "propagate": "#c0392b",
                            }
                            color = tier_colors.get(tier, "#95a5a6")
                            verified = entry.get("recovery_verified", False)
                            check = "Verified" if verified else "Not verified"
                            st.markdown(
                                f'<div style="border-left:4px solid {color}; padding:8px 12px; '
                                f'margin-bottom:8px; background:rgba(255,255,255,0.03); border-radius:0 6px 6px 0;">'
                                f'<strong>Round {entry.get("round", "?")}:</strong> '
                                f'<span style="color:{color}; font-weight:600;">{tier.upper()}</span><br>'
                                f'<span style="font-size:13px; color:#bbb;">'
                                f'Failed action: <code>{entry.get("failed_action", "N/A")}</code><br>'
                                f'Error: {entry.get("error", "N/A")}<br>'
                                f'New actions: {entry.get("n_new_actions", 0)} | {check}'
                                f'</span></div>',
                                unsafe_allow_html=True,
                            )

                # Extra metrics for ours
                om1, om2, om3 = st.columns(3)
                om1.metric("Recovery", r.get("recovery_attempts", 0))
                om2.metric("Switches", r.get("strategy_switches", 0))
                om3.metric("Rollbacks", r.get("rollbacks", 0))

    # ─── Step-by-step playback (Our System) ────────────────────────────────────

    if "ours" in results and results["ours"]["actions"]:
        st.markdown("---")
        st.subheader("Step-by-Step Playback (Our System)")

        ours_actions = results["ours"]["actions"]
        # Use the state from which actions_executed is actually valid
        playback_start = results["ours"].get("actions_start_state") or initial
        recovery_trace = results["ours"].get("recovery_trace", [])

        # Figure out which step indices were recovery-generated
        recovery_step_indices = set()
        for entry in recovery_trace:
            step_idx = entry.get("step_index", -1)
            n_new = entry.get("n_new_actions", 0)
            if entry.get("recovery_verified", False) and step_idx >= 0:
                for ri in range(step_idx, step_idx + n_new):
                    recovery_step_indices.add(ri)

        step = st.slider(
            "Step",
            min_value=-1,
            max_value=len(ours_actions) - 1,
            value=-1,
            format="Step %d",
            help="-1 = initial state, then 0..N-1 = after each action",
        )

        pcol1, pcol2 = st.columns([3, 2])

        with pcol1:
            if step == -1:
                render_blocks(playback_start, "Start State")
            else:
                sim_state, sim_holding, err_at, err_msg = simulate_step(playback_start, ours_actions, step)
                is_recovery = step in recovery_step_indices
                label = f"After step {step + 1}: {ours_actions[step]}"
                if is_recovery:
                    label += "  [RECOVERY]"
                if err_at is not None:
                    label = f"ERROR at step {err_at + 1}: {err_msg}"
                render_blocks(sim_state, label, holding=sim_holding)
                # Debug info
                with st.expander("Debug: simulation trace", expanded=False):
                    dbg_goal = BlocksWorldState(dict(playback_start.on))
                    dbg_env = BlocksWorldEnv(playback_start.copy(), dbg_goal)
                    dbg_env.reset()
                    for di, da in enumerate(ours_actions[:step+1]):
                        s, m = dbg_env.execute_action(da)
                        st.text(f"  {di+1}. {da} -> {'OK' if s else 'FAIL'}: {m}")
                        st.text(f"     State: {dbg_env.state.on}  Holding: {dbg_env.holding}")

        with pcol2:
            st.markdown("**Action Sequence**")
            for i, action in enumerate(ours_actions):
                is_recovery = i in recovery_step_indices
                if i == step:
                    marker = ">>  "
                    style = "font-weight:bold; color:#f1c40f;"
                elif i < step:
                    marker = "    "
                    style = "color:#888; text-decoration:line-through;" if False else "color:#2ecc71;"
                else:
                    marker = "    "
                    style = "color:#666;"

                prefix = ""
                if is_recovery:
                    prefix = '<span style="color:#e67e22; font-size:11px;">[R] </span>'

                if i == step:
                    st.markdown(
                        f'<div style="background:rgba(241,196,15,0.1); border-left:3px solid #f1c40f; '
                        f'padding:4px 8px; border-radius:0 4px 4px 0; margin:2px 0;">'
                        f'{prefix}<span style="{style}">{i+1}. {action}</span></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div style="padding:2px 8px; margin:1px 0;">'
                        f'{prefix}<span style="{style}">{i+1}. {action}</span></div>',
                        unsafe_allow_html=True,
                    )

        # Show goal comparison at current step
        if step >= 0:
            sim_state, _, _, _ = simulate_step(playback_start, ours_actions, step)
            gcol1, gcol2 = st.columns(2)
            with gcol1:
                st.markdown("**Current positions:**")
                for block in sorted(sim_state.on.keys()):
                    loc = sim_state.on[block]
                    goal_loc = goal.on.get(block, "?")
                    match = loc == goal_loc
                    icon = "[ok]" if match else "[--]"
                    st.markdown(f"`{icon}` **{block}**: on {loc}" + (f" *(goal: on {goal_loc})*" if not match else ""))
            with gcol2:
                render_blocks(goal, "Goal (reference)")
