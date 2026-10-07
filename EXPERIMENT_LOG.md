# Experiment log

## 2026-10-06: ReAct baseline run #1 was invalid (degenerate fixed point)

**Run:** `runs/20261006-234212_react_V2_task_01_tomato_seed0` (qwen2.5:7b via Ollama,
temperature 0.0, history 10, 1 action per call, 300 steps, one seed).

**What happened:** from call 3 to call 300 the model produced the same reply every turn
("pick up the tomato", `Action: 1`, the same bug report). It walked into the trash can at
(1,4) for 297 steps. Results: 0 bugs found, 0 deliveries, 299 identical false bug reports.

**Not a harness bug.** Each prompt showed the agent's true position (1,3), the event
"nothing happened (bumped trash_can)", and the tomato at (0,2).

**Root causes**
1. *Perception:* the 7B model misread the ASCII grid. It believed it was adjacent to the
   tomato, which was diagonal from it, and nothing in the observation listed the tiles it
   could actually reach in one move.
2. *Sampling:* at temperature 0 the same prompt gives the same reply. Once the 10-turn
   history filled with identical turns, the prompt barely changed, so the loop reinforced
   itself.
3. *Protocol:* one seed and no sanity check before an hour-long run. The failure only became
   visible after the run.

**Why it matters:** the run measures a stuck agent, not ReAct exploration. Comparing it with
random play (17 bugs in 10k steps, 8 in the first 300) would understate the LLM arm for a
reason unrelated to the research question.

**Fixes (kept within "vanilla ReAct": no planning, reflection, memory or loop-breaking hints)**
1. The observation gains an `ADJACENT` block: what lies one move away in each direction.
   This is perception only, and every arm uses the same renderer.
2. Default temperature 0.7, and runs use several seeds (`--seeds 0,1,2`).
3. Every configuration first gets a 50-step pilot that must pass `react_baseline.pilot_check`
   before any full run.

The run is kept as evidence of the failure mode. Do not use it as the baseline result.

## 2026-10-07: Pilot after fixes 1–3 failed the gate (full runs not started)

**Run:** `runs/pilot/20261007-101035_react_V2_task_01_tomato_seed0` (qwen2.5:7b, temperature 0.7,
`ADJACENT` block, 50 steps, seed 0).

**Gate:** FAIL. Distinct tiles visited: 3 (need ≥ 5). Passed: 11 pickups, longest
same-action streak 3, parse error rate 0%, replay matches the log. 1 bug found (19).
48 bug reports, all mistaken (for example, "moving down does not deliver the tomato").

**What happened:** after step 4 the agent repeated a 4-step cycle:
pick up the tomato → step down → move down into the trash can, which throws the tomato
away (it respawns) → step up. Its stated reasoning contradicts its actions and the
`ADJACENT` block. For example, it chose `Action: 3` with the thought "moving up should walk to
the tomato" while the observation listed the tomato as `2 left`.

**Interpretation:** the fixes changed the failure from a fixed point into a short cycle, but
qwen2.5:7b does not reliably map the stated directions to action IDs, even when the
observation spells them out. The remaining bottleneck is model capability, not
observation format or sampling. The gate criteria were not relaxed.
