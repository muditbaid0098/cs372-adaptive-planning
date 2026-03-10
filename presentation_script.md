# Presentation Script (~5 minutes)

---

## Slide 1: Title (5 sec)

"Hi, I'm Mudit. My project is on adaptive hierarchical planning with failure detection and strategy switching for LLM-based agents."

---

## Slide 2: Outline (5 sec)

"I'll cover the problem, our system design, evaluation results, and limitations. Let's jump in."

---

## Slide 3: LLMs as Planners -- Three Key Limitations (40 sec)

"So when you use an LLM as a planner, you hit three problems.

First, there's no hierarchy. The model generates a flat list of actions -- pick this up, put that down -- with no distinction between high-level goals and low-level steps. If step 3 is wrong, everything after it is probably wrong too, and there's no way to catch that.

Second, there's no strategy adaptation. A subtask that needs spatial reasoning gets the same prompting as one that needs logical deduction. We know CoT, ToT, and ReAct each have strengths, but no system picks the right one based on what the subtask actually needs.

Third, and this is the big one -- there's no failure recovery. If your plan breaks at step 5, current systems either restart from scratch or just keep going down the wrong path. All the progress from steps 1 through 4 is wasted."

---

## Slide 4: Our Approach -- Key Insight (30 sec)

"Our key insight is that if you decompose the task hierarchically, you can recover from failures without throwing away everything.

We combine four ideas: hierarchical decomposition from classical HTN planning, adaptive strategy selection across different reasoning methods, deterministic verification through simulation, and a tiered recovery system that tries cheap fixes first before expensive re-planning.

We evaluate on BlocksWorld -- it's a classical planning domain, fully deterministic, with well-defined action semantics. That gives us a ground-truth verifier, which is important for reliable evaluation."

---

## Slide 5: System Architecture (30 sec)

"Here's the architecture. Five modules in a pipeline with a feedback loop.

The forward pass goes left to right: a task comes in, the Decomposer breaks it into a subtask tree, the Classifier labels each subtask by type, and the Strategy Executor runs the appropriate reasoning strategy.

Then the Verifier simulates the plan against the actual BlocksWorld physics. If it passes, we're done. If it fails, the Failure Handler kicks in -- it can either switch to a different strategy, which is cheap, or roll all the way back to the Decomposer for a full re-plan, which is expensive but more powerful."

---

## Slide 6: Modules 1-3 -- Decompose, Classify, Execute (35 sec)

"The Decomposer uses GPT-4o to generate a hierarchical subtask tree in JSON. We cap it at depth 3 and branching factor 5 to prevent over-decomposition. One key feature is state-diff prompting -- instead of just giving the LLM the full state description, we compute a structured diff showing exactly which blocks need to move and which are already correct. This focuses the LLM's attention.

The Classifier labels each subtask as spatial, procedural, logical, and so on. It uses context from parent and sibling nodes for consistency.

Then the Strategy Executor maps each type to a reasoning strategy -- spatial tasks get state tracking, procedural gets precondition checking, logical gets Tree-of-Thought. When recovery triggers a strategy switch, there's a fallback ordering for each type."

---

## Slide 7: Modules 4-5 -- Verify and Recover (40 sec)

"The Verifier is purely deterministic -- it simulates every action against the BlocksWorld rules, checks preconditions, and verifies the goal is reached. No LLM involved, so no hallucination risk.

When a plan fails, the Failure Handler has four tiers, ordered by cost.

First, strategy switch -- just re-run the failed subtask with a different reasoning approach. Costs one LLM call.

Second, surgical repair -- keep the actions that worked, and ask the LLM to fix just the remaining steps using the failure diagnostics. The diagnostics include things like blocking chain analysis -- 'to move A, you first need to move C which is sitting on top of A.'

Third, rollback -- full re-decomposition from the current state with all the diagnostic context.

Fourth, propagate -- give up after 3 rounds. The idea is try cheap fixes first, only escalate when needed."

---

## Slide 8: Live Demo (30-60 sec)

"Before we get into the numbers, let me show you the system in action. I built a Streamlit app where you can generate any BlocksWorld task, run all the methods side-by-side, and step through the actions visually."

**[Play the demo video or switch to live demo]**

"So here I'm generating a task with [X] blocks. You can see the initial state on the left and the goal on the right. I'll run our system and Flat CoT together.

CoT fails -- it generates a plan that violates preconditions partway through. Our system also hits an error, but watch -- the recovery kicks in, switches strategy, and finds a valid plan.

Down here you can step through the actions one by one and see the blocks move. You can also see the decomposition tree and the recovery trace showing exactly which tier was used."

**[Stop video/demo]**

"Alright, let's look at the full evaluation."

---

## Slide 9: Experimental Setup (20 sec)

"We generated 60 BlocksWorld tasks: 20 easy with 3-4 blocks, 20 medium with 5-6, 20 hard with 7-8. Fixed seeds for reproducibility. All methods use GPT-4o.

Three baselines: Flat CoT which is single-pass reasoning, Flat ToT which generates 3 candidates, and ReAct which does up to 15 rounds of interleaved reasoning and acting. Plus six ablation variants to isolate each module's contribution."

---

## Slide 10: Main Results (30 sec)

"Here are the main results. Our system hits 82% overall, compared to 70% for CoT, 57% for ReAct, and 48% for ToT.

The key story is in the hard tasks column -- 80% versus 20 to 50%. On easy tasks we're comparable to CoT, which makes sense -- you don't need hierarchical planning for a 3-block problem. But as complexity grows, the gap opens up.

Pairwise McNemar's tests show significance against ToT at p less than 0.001 and ReAct at p less than 0.01. The gap versus CoT is not statistically significant on this benchmark alone, which brings us to the next slide."

---

## Slide 11: Scaling -- The Gap Widens (30 sec)

"To test scaling, we generated 20 additional tasks with 9 and 10 blocks -- well beyond the main benchmark.

The results here are dramatic. Our system hits 90% overall. Flat CoT drops to 30%. And ReAct gets zero percent -- its 15-round budget is simply not enough for problems that need 25 or 30 actions.

This is the strongest evidence that hierarchical planning with recovery actually scales, while flat approaches hit a wall. At this complexity, you need to decompose the problem and you need to recover from errors."

---

## Slide 12: Ablation Study (30 sec)

"The ablation tells us what actually matters. Removing the verifier and failure handler together drops performance by 25 points, down to 57%. That's the biggest single effect.

Removing just the failure handler drops 17 points to 65% -- so verification alone helps by filtering bad plans, but you need recovery to convert detected failures into successes.

The classifier and adaptive strategy play a supporting role. Removing the adaptive strategy hurts hard tasks by 5 points. Their contribution flows through the recovery mechanism -- strategy switching only matters when you're recovering from failure."

---

## Slide 13: Recovery Analysis (25 sec)

"Here's the recovery breakdown. 34 out of 60 tasks succeeded on the first try. 24 triggered recovery, and we recovered 15 of those -- 63% recovery rate.

Without recovery our system would be at 57%, which is about the same as ReAct. So recovery accounts for the entire 25-point gap over the best baseline.

What's interesting is hard tasks have the highest recovery rate at 73%. State-diff prompting and failure diagnostics are most useful when the task is complex enough to provide actionable guidance. Seven tasks were solved only by our system -- every baseline failed on all seven."

---

## Slide 14: Efficiency (20 sec)

"On efficiency -- our system uses 3.7 LLM calls per task on average, compared to ReAct's 11.3. We're nearly 6 times more token-efficient per successful task.

And we're actually faster than ReAct in wall-clock time -- 8.7 seconds versus 9.3 -- despite having a more complex pipeline and a higher success rate. The efficiency comes from targeted calls rather than iterative rounds."

---

## Slide 15: Failure Modes (25 sec)

"Of the 11 tasks we fail on, the dominant failure mode at 73% is cascading recovery errors. The initial plan fails, and then each recovery tier also generates a flawed plan. This is the main thing to improve.

There's an interesting pattern in the failure rates by block count -- 8-block tasks actually fail less than 6-block tasks. This suggests task structure matters more than raw complexity, and that our state-diff prompting is particularly effective on highly structured configurations.

On plan quality, our successful plans are 1.03x optimal length, versus 1.13 to 1.15x for baselines."

---

## Slide 16: Limitations (20 sec)

"Important limitations to be upfront about.

We only evaluate on BlocksWorld, which is deterministic and fully observable. The verifier won't directly transfer to domains with partial observability or stochastic dynamics.

Our difference versus CoT isn't statistically significant on the 60-task benchmark, though it is on the scaling experiment. Sample size is moderate.

And the system's dominant failure mode -- cascading recovery -- means that recovery can sometimes make things worse rather than better."

---

## Slide 17: Future Work (15 sec)

"For future work, the most interesting directions are extending to partially observable domains like ALFWorld, learning decomposition and strategy selection from experience rather than using fixed rules, and building meta-reasoning about when to stop recovery rather than always trying all three rounds."

---

## Slide 18: Conclusion (25 sec)

"To wrap up -- we built a modular planning system that combines hierarchical decomposition, adaptive strategies, deterministic verification, and tiered recovery.

82% on the main benchmark, 90% on scaling, versus 70% and 30% for the best baseline. The advantage is entirely driven by the recovery mechanism, which accounts for 25 extra percentage points.

The system is also efficient -- 3.7 calls per task, 6x more token-efficient than ReAct, and faster wall-clock time.

Thank you. Happy to take questions."

---

## Timing Summary

| Section | Slides | Time |
|---------|--------|------|
| Problem + Approach | 3-4 | ~70 sec |
| System Design | 5-7 | ~105 sec |
| Demo | 8 | ~30-60 sec |
| Results | 9-14 | ~155 sec |
| Failures + Limitations | 15-16 | ~45 sec |
| Future + Conclusion | 17-18 | ~40 sec |
| **Total** | **18** | **~5 min 45 sec** |

To trim to 5 min: shorten the demo to 30 sec (just play the video), cut the efficiency slide (14), or compress module details (6-7).
