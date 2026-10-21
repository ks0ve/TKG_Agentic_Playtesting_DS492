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

## 2026-10-20: Baseline v2: shared Action reader; react-raw kept as an ablation

**Problem 1, the strawman.** The MVP ReAct agent emits one raw move (0–4) per LLM call. Most
calls are spent on navigation: walking three tiles costs three LLM calls before anything
interesting happens. With qwen2.5:7b it also cannot reliably map directions to action IDs (see
2026-10-07), so it fails the pilot gate on navigation alone. A baseline that spends its budget
walking makes any memory-based arm look better for a reason unrelated to the research question.

**Problem 2, the action-interface confound.** The TKG arm will plan at a higher level (a cheap
policy for routine moves, the LLM only when memory flags an unexpected outcome). If that arm
also gets a coarser action interface than ReAct, the comparison mixes two effects: "memory
decides *when* to call the LLM" and "the action interface is coarser".

**Fix.** One shared, deterministic, LLM-free **Action reader** (`react_baseline/action_reader.py`)
that every agent uses: `goto(x,y)`, `interact(x,y)` (or a unique station/item name), `wait(n)`,
or a raw move, executed by BFS over floor tiles with fixed tie-breaks, early stops (episode end,
budget, unexpected event, 40-step cap), and one wait plus an agent-visible error for an invalid
intent. Raw moves from `react-raw` and `random` become one-step intents through the same reader,
so all four agents (`react`, `react-raw`, `random`, `random-intent`) share the runner, reader,
budgets and logging. The only difference left between arms is when the LLM is consulted.
`react` is consulted at every decision point (`--intents-per-call 1`), so it stays an always-on
ReAct baseline. The reader is the action interface, not a planner: it never decides what to do.
ReAct stays vanilla (rolling history only; the prompt names no defect).

**react-raw is kept as an ablation.** It sends byte-for-byte the MVP prompts (snapshot test), and
its legacy step fields are identical to the MVP runner's for the same seed (checked on 2000 random
and 300 mock-ReAct steps). It measures the cost of the raw interface.

**Bug taxonomy (draft).** `configs/bug_taxonomy_draft.csv` classifies all 165 SAGE bug IDs
(family, reachability, trigger signature, proposed RQ3 class; all `status=proposed`). Runs now
report `bugs_by_proposed_class` / `bugs_by_family`. This is **measured alongside, not yet used for
analysis**: it changes no filter or metric (`ignore_bug_ids` stays `[5]`). No run so far found a
bug the draft marks unreachable on shipped maps.

**Other changes.** Model registry (`configs/models.yaml`, `--model-profile`), an OpenAI-compatible
backend for self-hosted servers, `--max-usd` and USD/call-rate/latency metrics, structured
agent-visible transitions (`transitions.jsonl`), and strict bug-report attribution (a report at
decision d is checked against the oracle IDs fired by decision d−1) next to the legacy ±3 window.
The matched **env-step budget** is the primary comparison axis.

**Found while doing this.** The MVP Anthropic backend passed `temperature=` to
`messages.create`, which anthropic 1.11.0 does not accept (`TypeError` on the first call); the paid
path had never run. Temperature now goes through `extra_body`, and a test binds the request to the
installed SDK. Claude Haiku 5.5 rejects non-default temperatures, so its profile runs at the API
default; `haiku-4.5` keeps temperature 0.7 for a like-for-like comparison with local runs.

**Mock verification (V2_task_01_tomato, seed 0, offline).** 300 steps: `react` 192 decisions =
192 LLM calls (0.64 calls per env step), bugs [15, 20, 28]; `react-raw` 300 calls (1.0 per step),
8 bugs; `random-intent` 129 decisions, 3 bugs. 10,000 steps: `random` 17 bugs (same set as the
MVP), `random-intent` 15 bugs. Mock replies are random, so these numbers test the harness, not
the agents. Same seed gives byte-identical `steps.jsonl` and `transitions.jsonl` for all four
agents.

## 2026-10-20: qwen2.5:7b pilots on v2: both action spaces fail the gate (no full runs started)

V2_task_01_tomato, seed 0, 50 env steps, temperature 0.7, history 10, `num_ctx` 8192, Ollama 0.40.0
on the RTX 3050 laptop.

| | `react` (intents) | `react-raw` (raw moves) |
|---|---|---|
| run | `runs/pilot/20261020-164743_react_…` | `runs/pilot/20261020-171750_react-raw_…` |
| gate | **FAIL** (tiles 4 < 5; intent failures 75% > 30%) | **FAIL** (tiles 3 < 5) |
| passed | 1 pickup, longest identical-intent run 10, parse errors 2.1%, replay exact | 11 pickups, longest same-action streak 3, parse errors 0%, replay exact |
| decisions = LLM calls | 48 (0.96 per env step) | 50 (1.0 per env step) |
| tokens per call (in / out) | 2,227 / 87 | 1,614 / 45 |
| median latency per call | 19.6 s (one 790 s outlier while CPU-heavy tests ran in parallel) | 9.7 s |
| bugs | none | [19] |
| bug reports (strict / lenient match) | 12 | 48 (2 / 14) |

**What `react` did.** Every one of the 36 failed intents was the same confusion between the two
navigation intents: `goto` aimed at a counter or station (25 × "not a floor tile", e.g. 10 calls
in a row of `goto(0,2)` at the tomato) or `interact` aimed at a floor tile (11 × "(2,3) is a floor
tile; use goto"). Each error reached the model in the next prompt via the executor trace; it
often repeated the intent anyway and filed bug reports blaming the game ("repeated failure to go
to (0,2) despite it being a floor tile"). Because every failed intent costs one wait step, the
call rate stayed near one call per env step instead of the ~0.6 the mock reached.

**What `react-raw` did.** The same short pick-up / walk / trash cycle as on 2026-10-07: 11 pickups
within 3 tiles, 48 mistaken bug reports.

**Interpretation.** The Action reader removed the direction-to-ID mapping problem, but qwen2.5:7b
does not follow the intent grammar's floor / non-floor distinction even with explicit error
feedback. This is a capability limit of the 7B model, not a harness problem (replay exact, parse
errors low, intent solutions for all 6 demo tasks pass in tests). The gate was not relaxed and
the prompt was not tuned toward this map. Per protocol, no full runs were started. Next: pilot a
stronger model through the same interface (Haiku 4.5 / 5.5, or qwen2.5:14b), e.g.
`--model-profile haiku-4.5 --pilot --max-usd 0.25`.

## 2026-10-20: qwen2.5:14b pilots: both fail, but `react` gets much closer

Same settings as the 7b pilots (tomato, seed 0, 50 steps, temperature 0.7, history 10, `num_ctx`
8192). The 9 GB model does not fit the 4 GB GPU, so calls are slow.

| | `react` (intents) | `react-raw` (raw moves) |
|---|---|---|
| run | `runs/pilot/20261020-191343_react_…` | `runs/pilot/20261020-194257_react-raw_…` |
| gate | **FAIL** (intent failures 48.6% > 30%) | **FAIL** (same-action streak 14 > 10) |
| passed | tiles 8, pickups 4, longest identical-intent run 6, parse errors 0%, replay exact | tiles 6, pickups 1, parse errors 0%, replay exact |
| decisions = LLM calls | 35 (0.70 per env step) | 50 (1.0) |
| tokens per call (in / out) | 2,164 / 68 | 2,092 / 107 |
| median latency per call | 48 s | 67 s |
| deliveries | **1** (order completed) | 0 |
| bugs / bug reports | none / 4 (0 strict, 0 lenient matches) | none / 35 (0 / 0) |

**`react`.** All 17 failed intents were again the floor / non-floor confusion (15 × `goto` on a
station or counter, 2 × `interact` on a floor tile), as were all 36 of the 7b's. Unlike the 7b,
the 14b usually corrected itself after the error (e.g. `goto(3,0)` ×7, then `interact(3,0)`
worked) and completed the order. **`react-raw`.** It walked right into the wall for 14 steps in a
row (36 of 50 actions were `right`).

## 2026-10-20: Action reader tolerates goto / interact floor mix-ups (`a53743d`)

**Why.** Every failed intent in the first v2 pilots was the same confusion between the two
navigation verbs (qwen2.5:7b 36/36, qwen2.5:14b 17/17): `goto` aimed at a counter or station
("not a floor tile") or `interact` aimed at a floor tile. Each cost an LLM call plus a wasted wait
step even though the intended target was clear.

**Change.** `goto(x,y)` on a non-floor tile now walks to the nearest floor tile next to it and stops
**without bumping** (it never performs an interaction the agent did not ask for); `interact(x,y)`
on a floor tile walks onto it. Both are noted in the agent-visible trace and logged as `coerced`
(`executor_result`, `transitions.jsonl`, summary `intents_coerced` / `intent_coercion_rate`); they
are not intent failures. Off-map, unreachable and unparseable intents are still errors. The prompt
and the pilot gate are unchanged.

**Scope.** This is part of the shared Action reader, so it applies identically to every agent,
including the future TKG arm's cheap policy and LLM. It is interface design, not planning.

## 2026-10-20/21: Post-tolerance pilots (tomato, seed 0, 50 steps, temperature 0.7, history 10)

| model | agent | run (`runs/pilot/…`) | gate | calls/step | tok/call | median s/call | deliveries | bugs | reports (strict / lenient) |
|---|---|---|---|---|---|---|---|---|---|
| qwen2.5:7b | react | `20261020-205935_react_…` | FAIL: 2 tiles | 1.00 | 2,315 | 20.9 | 0 | – | 22 (0 / 0) |
| **llama3.1:8b** | react | `20261020-211948_react_…` | **PASS** | 0.48 | 1,928 | 8.2 | 1 | – | 0 |
| llama3.1:8b | react-raw | `20261020-212427_react-raw_…` | FAIL: 3 tiles | 1.00 | 1,838 | 18.2 | 0 | – | 0 |
| mistral-nemo:12b | react | `20261020-213913_react_…` | **PASS** | 0.48 | 2,087 | 20.1 | 0 | 19, 20 | 5 (0 / 1) |
| mistral-nemo:12b | react-raw | `20261020-215919_react-raw_…` | FAIL: same-action streak 14 | 1.00 | 1,932 | 23.1 | 0 | 20 | 17 (0 / 3) |
| qwen2.5:14b | react | `20261021-004127_react_…` | **PASS (barely)** | 0.70 | 2,437 | 29.1 (spikes ~190) | 1 | – | 3 (0 / 0) |

- **qwen2.5:7b**: 0% intent failures after the change (28/50 intents remapped), but it looped
  `goto(2,2)` ↔ `interact(2,2)` at the delivery window, trying to put a bare tomato on it
  ("putdown delivery failed") 21 times.
- **llama3.1:8b**: completed the order in 8 clean intents, then tried unusual interactions.
- **qwen2.5:14b**: rerun by the user (the 2026-10-20 checkpoint listed it as on hold). Passed with
  the longest identical-intent run at 9 against a limit of 10: its last 9 calls were all
  `interact(0,2)`, juggling the tomato. 1,340 s wall time for 50 steps.
- `react-raw` fails with every local model; as the ablation it is the result, so it runs ungated.

**Decisions (2026-10-21).** Headline LLM: Claude Haiku 5.5 (paid; $20 project budget, guarded by a
spend ledger). Local replication: llama3.1:8b. qwen2.5:14b is too slow for full runs and stays a
pilot data point. Grid: 1,000 steps per run on 3 tasks; Haiku 5 seeds (the API has no sampling
seed), local models 3 seeds; `react-raw` runs ungated as the ablation.

## Note: how SAGE's bug predicates behave

- The bug checks have **no side effects**: the game never actually misbehaves. A predicate fires
  when the agent *attempts* an edge-case interaction, so the oracle labels the agent's attempts,
  not visible malfunctions. An agent can therefore trigger a "bug" without seeing anything wrong.
- 52 of the 165 bug IDs are unreachable on the shipped maps (`unreachable_on_shipped_maps` in the
  draft taxonomy).
- The draft taxonomy (`configs/bug_taxonomy_draft.csv`) is reporting-only: it never filters
  results or changes metrics.

## 2026-10-21: Grid task selection (offline evidence; no LLM calls)

10,000 steps, seed 0, `ignore_bug_ids=[5]`; "excl. normal" drops IDs that also fire during the
scripted correct-play solution (`configs/normal_play_bugs.yaml`). All four tasks are completable
through the Action reader (scripted solutions: 17 / 34 / 41 / 57 steps).

| task | random: bugs (excl. normal) | random-intent: bugs (excl. normal) | plate_composition (random / r-intent) | IDs firing in correct play |
|---|---|---|---|---|
| 01 tomato | 17 (17) | 15 (15) | 3 / 1 | 5 |
| 07 tomato + steak | 16 (14) | 15 (14) | 2 / 1 | 1, 3, 5, 6, 21, 68, 127 |
| 11 tomato + lettuce + onion | 18 (16) | 12 (12) | **6** / 0 | 1, 2, 3, 5, 7, 66, 85, 86, 105, 133, 146 |
| 78 onion + fish + rice | 18 (16) | **23 (21)** | 3 / **7** | 1, 3, 5, 7, 26, 85, 135 |

**Chosen (as proposed, all check out):** 01 tomato (single ingredient; continuity with every
pilot), 07 tomato + steak (adds the pan and cooking over time), 11 tomato + lettuce + onion (three
ingredients; reaches the deep plate-combination rules: 6 plate_composition bugs under random play).
**Alternative:** 78 onion + fish + rice reaches the broadest coverage (rice cooker, sushi merge,
23 bugs and one `PENDING_directed_check` bug under random-intent); swapping it in for 11 is one
line in `configs/grids/baseline_v2.yaml` (re-freeze the task list too). Correct play itself fires
up to 11 oracle IDs on multi-ingredient tasks, so `unique_bugs_excluding_normal_play` matters
there; the normal-play sets depend on the scripted path and are reporting-only.
