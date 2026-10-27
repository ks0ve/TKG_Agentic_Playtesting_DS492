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
- 51 of the 165 bug IDs are unreachable on the shipped maps (`unreachable_on_shipped_maps` in the
  draft taxonomy; 52 until 2026-10-26, when bug 78 was corrected to `fires_in_runs`).
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

## 2026-10-21: Event text used SAGE's internal names (translator bug, fixed)

**Haiku 5.5 pilot** (`runs/pilot/20261021-210802_react_…`, tomato, seed 0, 50 steps, repro on with
15-step attempts): 26 calls, $0.013 (ledgered), 1 reproduction attempt. Both of its bug reports
were **false and caused by our label**, not the game: after putting the tomato on the cutting
board the event read "putdown knife succeeded", and Haiku reported that a knife was being
mentioned where none exists. (Local llama3.1:8b pilot `20261021-140329_react_…` with repro on:
PASSED, 0 reports, so no attempt.)

**Fix.** Event text (`describe_event` → LAST EVENT, history summaries, executor traces,
transitions, `visible_state.last_event.target`) now uses the same display names as the map
legend and STATIONS list: knife → cutting board, trash_can → trash can, delivery → delivery
window, ricecooker → rice cooker, pan → pan (stove), block → wall. Structured `stations[].kind`
keeps the internal id. A test walks random valid intents on all 37 task maps and fails on any
internal name in agent-visible text. The renderer is shared by every agent; the MVP prompt
snapshot changed in exactly one line ("bumped delivery" → "bumped delivery window"). The earlier
pilots ran with the old labels.

## 2026-10-26: Translator wording v2.1 (event text from the player's view)

**Why.** In the 33-run v2.0 progress grid, Haiku filed 1,019 bug reports, and about 73% were about
our text interface, not the game:
- "putdown plate succeeded" was ambiguous;
- held items were listed twice, once with a coordinate;
- history step counters looked out of sync across macros, resets and repro;
- station letters did not show what a station holds.

**Change (shared translator, so `react`, `react-raw`, the random baselines and the TKG arm all get it).**
- **Event text** says what happened from the player's view: what you held before and after, the
  tile you acted on (legend name and position), and the result. It never explains why. Examples:
  - "put chopped lettuce onto the plate at (3,0); the plate now holds [chopped tomato, chopped lettuce]"
  - "tried to put tomato down at (1,4), where the onion is: nothing changed"
  - "tried to add from the empty cutting board at (0,3) to the plate you are holding: nothing changed"
  - "delivered the empty plate: order REJECTED"
  - "waited"

  When ingredients merge on a plate, the merged result is shown ("the plate now holds [sushi]").
  The full approved table is in `react_baseline/tests/test_event_wording.py`.
- **Names.** The legend's display names are used everywhere, with no synonyms. The pan's legend
  entry is now just "pan" (was "pan (stove)"). The no-raw-names test covers all 37 maps and every
  channel: observation, history summary, event record text and names, executor trace, and full
  ReAct prompts. It also rejects SAGE event types ("putdown", "pickup", "dump"), "succeeded" and
  snake_case identifiers.
- **ITEMS** lists a held item as `- plate [clean; contents: ...] (held by you)`, with no coordinate.
- **Legend.** One added line: "Station letters do not show what the station holds; see STATIONS."
- **History lines** are labelled `[turn 12 | ep 3 | steps 14–18 (5 steps)]`, using episode steps
  numbered like the observation's `STEP: k/200` line. A test checks every label against
  steps.jsonl across macros, a natural 200-step episode end and a reproduction attempt.
- **Markers.** `=== EPISODE RESET: the kitchen was reset to its starting state ===`, and the same
  suffix on the reproduction start and end markers.
- **New event-record fields** (add-only; `type`, `target`, `success` unchanged), in
  transitions.jsonl and `visible_state.last_event`:
  - `held_before`, `held_after`
  - `tile`: the [x, y] the action was aimed at
  - `tile_before`, `tile_after`: `{kind, holding}`
- The system prompts are byte-identical. The react-raw prompt snapshot was regenerated, since its
  observations and history use the shared translator. The frozen identity changed; see
  `configs/frozen/baseline_v2_1.yaml`.

**Token cost.** The longer event text and labels increase prompt size slightly. Measured on mock
runs (300 steps, seed 0; identical action sequences before and after; mean characters per call):

| arm / task | user prompt before → after | system + user |
|---|---|---|
| react / 01 tomato | 3,522 → 3,896 (+10.6%) | 6,401 → 6,775 (+5.8%) |
| react / 07 tomato+steak | 3,558 → 3,945 (+10.9%) | 6,437 → 6,824 (+6.0%) |
| react / 11 salad | 3,514 → 3,874 (+10.3%) | 6,392 → 6,753 (+5.6%) |
| react-raw / 01 tomato | 2,368 → 2,765 (+16.8%) | 4,686 → 5,083 (+8.5%) |
| react-raw / 07 tomato+steak | 2,401 → 2,768 (+15.3%) | 4,719 → 5,086 (+7.8%) |
| react-raw / 11 salad | 2,379 → 2,763 (+16.1%) | 4,697 → 5,081 (+8.2%) |

That is about +6% (react) and +8% (react-raw) input tokens per call. Mock agents rarely reach
plates or stations, so event-heavy real runs may grow somewhat more.

## 2026-10-26: Order-independent normal-play bug sets (recipe planner)

**Why.** The old normal-play sets came from one scripted solution per task, so they reflected one
recipe order. For example, bug 78 (a steak on the plate, then a tomato added) fired in the v2.0
grid but never on the scripted path, which adds tomato first.

**Planner** (`python -m react_baseline.normal_play --task <id> | --grid | --all`, offline). It plays
every correct-play variant on the real game through the shared Action reader:
- **Preparation order:** interleaved (prepare an ingredient, add it, prepare the next), or
  prepare-first in every order (chopped vegetables parked on counters; a steak keeps cooking in
  the pan meanwhile).
- **Assembly order:** every order.
- **Plating method per ingredient:**
  - scoop into the held plate from the cutting board, pan or a counter;
  - carry the item and put it down onto the plate on a counter;
  - park it on a counter, then scoop it.
- **Plate handling:** the plate is held between consecutive scoops or put down partway.

Plates always start clean, and a correct delivery ends the episode (SAGE `Plate.__init__` /
`overcooked_env.step`), so no correct path washes a plate. There is no washing variant.

Oracle IDs are recorded on every step whose event did not fail. No valid variant has a failed
event. A variant counts only if the delivery is accepted.

`configs/normal_play_bugs.yaml` now has, per task:
- the union (`bug_ids_raw`, used by the metrics)
- the variant count
- the IDs that fire in every variant
- how many variants fire each ID
- the full per-variant breakdown
- the scripted solution's IDs, for comparison

Task 78 (not in the grid) keeps its scripted-solution set. The taxonomy draft row 78 now reads
`fires_in_runs` / `EXCLUDED_normal_play` (copied from the root draft into
`configs/bug_taxonomy_draft.csv`).

| task | variants (all delivered) | normal-play union | in every variant | old (scripted) set |
|---|---|---|---|---|
| 01 tomato | 5 | 5 | 5 | 5 |
| 07 tomato + steak | 46 | 1, 3, 4, 5, 6, 21, 68, 78, 127 | 5, 6 | 1, 3, 5, 6, 21, 68, 127 |
| 11 tomato + lettuce + onion | 558 | 1, 2, 3, 4, 5, 7, 66, 67, 70, 71, 73, 74, 85, 86, 87, 88, 102, 103, 105, 107, 109, 133, 134, 135, 146, 147, 148 | 5, 7 | 1, 2, 3, 5, 7, 66, 85, 86, 105, 133, 146 |

**Cross-check against the hand-built lists from the v2.0 runs.**

The hand-built lists were:
- tomato: 47
- tomato + steak: 4, 6, 68, 78, 127
- salad: 3, 4, 7, 70, 73, 74, 86, 87, 88, 102, 103, 109
- uncertain: 67, 71, 107

In the planner but not in the hand-built lists:
- **IDs that fire in every variant:** 5 (every completed chop; already in `ignore_bug_ids`), 6 and
  7 (delivering two or three items).
- **Plate handling:**
  - 1 (adding to a held plate that holds one item): any scoop as the second ingredient.
  - 3 (putting a plate with one item on a counter): putting the plate down partway.
- **Timing:** 21 (cooked steak stays in the pan for a step): any variant that does not take the
  steak out on the step it finishes cooking.
- **Cutting-board scoops:** 2, 85, 105, 133, 134, 135, 146, 147, 148. These are scooping a chopped
  vegetable from the cutting board into a plate that already holds one or two items. They fire in
  18–54 of the 558 salad variants.
- **Salad orders:** 66, 67, 71, 107. The ordered pairs (tomato→lettuce, tomato→onion,
  lettuce→onion, tomato→chopped onion) each fire in 93 variants. They are as normal as their
  mirror images (70, 73, 74, 109), which were in the hand-built list. This **resolves 67, 71 and
  107 as normal play.**

In the hand-built lists but not in the planner:
- **47** (SINK_HOLD_AFTER_CLEAN, tomato). It fires when a washed plate sits clean in the sink.
  That needs a dirty plate, and the only source is a delivery. A correct delivery ends the
  episode, so 47 comes from recovering after a REJECTED delivery (wash the returned plate, then
  deliver again). That is not correct play from the starting state, so it stays a counted bug.
  Say if you want recovery play treated as normal.

Every other hand-built ID is in the planner's union: 4, 6, 68, 78, 127 for tomato + steak; all 12
for the salad.

Variants that never put the plate down (carry every item onto the plate on its counter) avoid
1 and 3. Variants that never scoop from the cutting board avoid the knife family. Every path
fires the delivery-count IDs.

`--all` (offline, about 6.5 min) plans 25 of the 37 tasks, all variants delivered except on 78.
The 12 skipped tasks need cheese or dough, which the planner does not handle. On 78 (onion + fish
+ rice), 240 of 726 variants fail once fish and rice merge into sushi: later additions to that
plate are refused. The file therefore keeps 78's scripted-solution set, and only the three grid
tasks are written.

## 2026-10-26: Adjusted metrics against the normal-play sets (v2.0 grid rescored)

`python -m react_baseline.rescore <run_dir|grid_dir>` writes `summary_rescored.json` next to each
`summary.json`, which is never modified. The file holds the original summary plus added fields,
with the task's normal-play IDs removed:
- `unique_bugs_excluding_normal_play` (new sets)
- `reports_strict_match_adj`, `unique_bugs_reported_strict_adj`
- per repro attempt: `source_bug_ids_strict_adj`, `oracle_reproduced_adj`
- `repro_success_rate_adj`, over attempts whose source report still has an ID
- `bugs_by_proposed_class_adj`

Nothing is re-run, and the oracle and `ignore_bug_ids` are unchanged.

v2.0 progress grid, means over 3 seeds (`runs/grids/baseline_v2/`):

| arm | task | bugs raw | bugs adj | deliveries | LLM calls | $ | reports strict adj | repro success adj |
|---|---|---|---|---|---|---|---|---|
| react / haiku-5.5 | 01 tomato | 1.0 | 1.0 | 55.3 | 879 | 0.492 | 0 | - |
| react / haiku-5.5 | 07 tomato+steak | 8.7 | 1.0 | 25.0 | 896 | 0.521 | 0.3 | - |
| react / haiku-5.5 | 11 salad | 16.0 | 2.7 | 19.7 | 800 | 0.462 | 0.3 | - |
| react-raw / haiku-5.5 | 01 tomato | 0.3 | 0.3 | 57.7 | 1000 | 0.513 | 0 | - |
| react / llama3.1-8b | 01 tomato | 10.7 | 10.7 | 1.0 | 757 | 0 | 11.3 | 0.75 |
| random | 01 / 07 / 11 | 8.3 / 10.3 / 9.0 | 8.3 / 9.7 / 9.0 | 0 | 0 | 0 | 0 | - |
| random-intent | 01 / 07 / 11 | 7.3 / 8.0 / 5.3 | 7.3 / 7.3 / 5.3 | 0 | 0 | 0 | 0 | - |

**Reading.** Haiku plays the game: 20–58 accepted deliveries per 1,000 steps. Almost everything
its runs trigger on the multi-ingredient tasks is normal play: 8.7 → 1.0 bugs on 07 and
16.0 → 2.7 on 11. Its strict-matching reports drop from 11–22 per run to 0–1. None of the 60
Haiku reproduction attempts (react and react-raw) had a source ID left after adjustment, so its v2.0 repro success rates
(up to 1.0) were all about normal-play IDs. llama3.1:8b and the random baselines rarely complete
a recipe, so adjustment barely changes them. These are v2.0 numbers, superseded for reports and
reproduction by v2.1 (next entry).

## 2026-10-26: v2.0 → v2.1

**What changed.** Only the agent-visible text changed. Settings, system prompts, the
Action reader (`2.1-floor-tolerant`), tasks, seeds and budgets are identical to v2.0.
- Translator v2.1 (entry above): player-view event text, legend names only, held items without
  coordinates, the station-letter legend note, history labels with episode-step ranges, and
  reset/reproduction markers that say the kitchen was reset.
- New add-only event-record fields: `held_before`, `held_after`, `tile`, `tile_before`,
  `tile_after`.
- Prompt size: about +6% (react) and +8% (react-raw) input tokens per call.

**Configs.**
- `configs/frozen/baseline_v2_1.yaml`: new prompt and renderer hashes. The prompt hash now also
  covers the episode-reset marker.
- `configs/grids/baseline_v2_1.yaml`: the full grid.
- `configs/grids/baseline_v2_1_progress.yaml`: the same 33 combinations as the v2.0 progress
  grid. Both write to `runs/grids/baseline_v2_1/`.
- The v2.0 files (`configs/frozen/baseline_v2.yaml`, `configs/grids/baseline_v2*.yaml`) stay as
  the record of `runs/grids/baseline_v2/`, marked superseded. They no longer pass the frozen check,
  by design.

**Normal-play sets.** These are the planner unions from the order-independent normal-play entry
above, used by `unique_bugs_excluding_normal_play` (new runs) and by the `*_adj` metrics:

| task | set |
|---|---|
| tomato | 5 |
| tomato + steak | 1, 3, 4, 5, 6, 21, 68, 78, 127 |
| salad | 1, 2, 3, 4, 5, 7, 66, 67, 70, 71, 73, 74, 85, 86, 87, 88, 102, 103, 105, 107, 109, 133, 134, 135, 146, 147, 148 |

`ignore_bug_ids` stays `[5]`. The oracle is unchanged.

**v2.0 results** (`runs/grids/baseline_v2/`, 33 runs) are kept and not re-run. Raw bug counts,
deliveries, calls and cost stay valid as v2.0 measurements, and `summary_rescored.json` adds the
adjusted metrics. The v2.0 results are **superseded for bug reports and reproduction metrics**:
about 73% of Haiku's reports were about our text interface, and its reproduction attempts
targeted normal-play IDs. Report-based comparisons should use v2.1 runs.

**2026-10-26, v2.1 addendum (before re-freezing).** The v2.1 Haiku salad pilot
(`runs/pilot/20261026-201139_react_V2_task_11_…`) cut false reports by roughly 4–5x compared with
v2.0. This is an estimate from a single 150-step pilot: 13 reports were expected at v2.0 rates,
and 3 were observed. Some remaining "reports" were non-reports: "Unconfirmed. …", or "… so no clear defect is
confirmed here." Two changes:
- OUTPUT FORMAT now says: "Only add a Bug report line when you believe you observed a defect.
  Never use it to say that no defect was found or that an earlier report is uncertain." This is
  format wording only, with no defect hints.
- Non-reports are filtered. A line starting with "unconfirmed", "no defect", "no bug", "not a bug",
  "nothing unusual", "nothing inconsistent", "nothing to report" or "nothing wrong" (a bare
  "nothing" was already ignored; "Nothing happened when ..." still counts), or containing "no (clear) defect/bug (is) confirmed/found/observed", is not
  counted as a report and never starts a reproduction attempt. It is logged in llm_calls
  `bug_report_discarded` and counted in summary `bug_reports_discarded`.

Prompt hash updated in `configs/frozen/baseline_v2_1.yaml`; the react-raw snapshot was regenerated.

## 2026-10-26: TKG arm v1 (branch `tkg-arm`; offline, mock LLM only)

Built on baseline-v2, porting the ideas of Tom's `origin/cheap-agent-baseline` (observed graph,
novelty agent) without merging it. The changes from his draft:
- **Structured data, not text.** The graph is built from `visible_state` and the event records,
  not observation text.
- **One transition per executed intent**, not per button press.
- **Abstract expectation keys.** Expectations are keyed on coordinate-free ctx0 / ctx1
  abstractions; exact-state hashes are kept only as the State node ids.
- **Intents.** The cheap policy emits intents through the shared Action reader, with novelty
  counts from the TKG.
- **No observation text in steps.jsonl.** transitions.jsonl already holds the structured states.

Modules: `memory/abstraction.py`, `memory/tkg.py`, `memory/trigger.py`,
`react_baseline/cheap_policy.py`, `react_baseline/tkg_agent.py`, plus add-only runner hooks.
See the README section "TKG arm".

**Mock measurements** (1,000 steps, seeds 0–2, tkg1 defaults: warm-up 200, k 3, tau 0.15,
min_support 3; mock LLM = seeded random intents):

| task | trigger fire rate (phase 2) | escalations / 1,000 post-warm-up steps | LLM calls / env step | fire reasons |
|---|---|---|---|---|
| 01 tomato | 0.256–0.324 (mean 0.29) | 70–81 (mean 77) | 0.23–0.28 | novel 349, surprising 2 |
| 11 salad | 0.290–0.305 (mean 0.30) | 69–78 (mean 74) | 0.23–0.25 | novel 343 |

**Cheap alone** (no LLM; 1,000 steps × 3 seeds), adjusted unique bugs per run (normal-play IDs
removed):

| task | cheap | random-intent (v2.0) | random (v2.0) |
|---|---|---|---|
| 01 tomato | 15.0 | 7.3 | 8.3 |
| 07 tomato + steak | 16.0 | 7.3 | 9.7 |
| 11 salad | 16.0 | 5.3 | 9.0 |

**Design points that did not work as written, or needed a decision:**
- **Novelty search drives the novelty trigger.** Almost every fire is "novel" (692 of 694). The
  cheap policy deliberately seeks situations with the lowest TKG count, often zero, so its own
  successes trip the novelty check. The mock LLM's random intents add more. "Surprising" almost
  never fires on mock runs. I kept the design and am flagging it; it is the first thing to look at
  in the Haiku pilot.
- **"≤ k decisions" is per fire.** A new fire during an investigation refreshes k without counting
  a new escalation, so the LLM keeps control while each of its own transitions deviates.
- **Memory check during reproduction.** It is still run and logged (`deviation`), but it is not
  counted in the fire rate and never escalates, since the LLM is already in control.
- **Warm-up boundary.** The check starts with the first intent that begins after 200 env steps;
  an intent straddling step 200 counts as warm-up.
- **Where "resume" is explained.** The system prompt is unchanged, so the "reply Action: resume"
  instruction lives in the MEMORY block. A resume reply is still an accounted LLM call, and the
  cheap policy acts in that same decision.
- **Pilot config.** There is no YAML pilot config: grid files are out of scope, and the grid
  driver does not know the TKG flags. The pilot command is in RUN_COMMANDS.

**2026-10-26, haiku-5.5 `max_output_tokens` 384 → 512** (all agents: react, react-raw, tkg). The
Haiku TKG pilot (`runs/pilot/20261026-225747_tkg_V2_task_11_…`) had 2 parse errors from replies
cut off before the Action line. The output cap only costs when it is used ($0.50/MTok), so 128
more tokens add at most $0.000064 per call. The v2.1 frozen config's profile note is updated; the
frozen hashes are unchanged, since the profile is not part of them. v2.1 runs made before this
date used 384.
