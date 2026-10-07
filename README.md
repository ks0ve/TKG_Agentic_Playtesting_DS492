# TKG Agentic Playtesting: ReAct baseline

The baseline arm of a study that compares LLM-agent exploration with algorithmic baselines
for game bug discovery. The game is **Overcooked Plus** from
[SAGE](https://github.com/BlueLinkX/SAGE). The package `react_baseline/` contains:

- a **vanilla ReAct agent** (Thought → Action → Observation). Its only memory is a rolling
  window of the last N turns. It has no planner, reflection, skill library, graph memory or
  pathfinding macros. Those belong to later experimental arms.
- a **random agent** (uniform over the 5 actions, seeded) that runs through the same harness
  and logging.
- a harness with budgets and auto-reset. It also keeps a **hidden oracle**: the env's
  bug-predicate IDs are logged but never shown to the agent.

```
react_baseline/
  env_adapter.py   make_env, render_text, EnvAdapter.step -> (agent-visible text, hidden oracle)
  prompts.py       system prompt + per-turn template
  llm.py           Ollama / Anthropic / Mock backends with token accounting
  agents.py        ReActAgent, RandomAgent, tolerant response parser
  runner.py        budgeted loop, auto-reset, all log files
  run.py           CLI
  pilot_check.py   pass/fail gate for a short pilot run
  tests/           pytest (offline)
```

## Setup

You need **Python 3.10 with tkinter**. The env's `debug_manager.py` runs `import tkinter`
even in headless mode. If tkinter is missing, `react_baseline` fails at import with a message
explaining how to fix it. tkinter is not stubbed.

- Windows/macOS: the python.org installers and `uv python install 3.10` both include Tk.
- Debian/Ubuntu: `sudo apt install python3.10-tk`.
- Check: `python -c "import tkinter; print(tkinter.TkVersion)"`

```sh
# 0. this repo
git clone https://github.com/ks0ve/TKG_Agentic_Playtesting_DS492.git
cd TKG_Agentic_Playtesting_DS492

# 1. the env (no license, so it is cloned locally and gitignored, never committed).
#    Check out the pinned commit so everyone runs identical env code.
git clone https://github.com/BlueLinkX/SAGE external/SAGE
git -C external/SAGE checkout 24de803f33123d16fe6f7cfa0062f3c81217a978

# 2. a Python 3.10 venv (uv shown; plain venv + pip also works)
uv venv --python 3.10 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt   # Windows
# uv pip install --python .venv/bin/python -r requirements.txt         # Linux/macOS

#    If uv fails with "incompatible hardlinks" (OneDrive folders), add --link-mode=copy.

# 3. tests (all offline, about 7 s)
.venv/Scripts/python.exe -m pytest
```

`requirements.txt` pins the SAGE env dependencies (numpy 2.2.6, gym 0.26.2, pygame 2.6.1,
PyYAML 6.0, networkx 3.4.2) and adds `anthropic`, `ollama` and `pytest`. The adapter adds
`external/SAGE/scenarios/overcooked_plus` to `sys.path`. It also sets `SDL_VIDEODRIVER`,
`SDL_AUDIODRIVER` and `PYGAME_HIDE_SUPPORT_PROMPT`, and silences the gym/NumPy-2 deprecation
banner.

## Running for free (Ollama, the default)

The default LLM backend is a local Ollama server. Nothing calls a paid API unless you pass
`--llm anthropic`.

1. Install Ollama from https://ollama.com/download.
2. Start the server if it is not already running: `ollama serve`
3. Pull the default model (4.7 GB): `ollama pull qwen2.5:7b`
4. Run:

```sh
python -m react_baseline.run --agent react --llm ollama --task V2_task_01_tomato --max-env-steps 200
```

If the server is unreachable or the model has not been downloaded, the CLI exits with code 2.
The message names the exact command to run (`ollama serve` or `ollama pull <model>`).

The Ollama call is `ollama.chat(model, messages, options={temperature, num_ctx, seed,
num_predict})`. The defaults are `--model qwen2.5:7b` and `--num-ctx 8192`. `num_predict` is
set from `--max-output-tokens` (default 512). Token accounting uses the response's
`prompt_eval_count` (input) and `eval_count` (output), so the logs have the same schema as the
Anthropic backend.

**Context-overflow guard.** Before each call the agent estimates the prompt size as
characters / 3.5. If the estimate is above 90% of `num_ctx`, it prints a
`[context guard]` warning and drops the oldest history turns until the prompt fits. If even a
prompt with no history does not fit, the run stops with an error asking for a larger
`--num-ctx`. The system prompt is never cut, so Ollama never has to truncate it silently.

## Other ways to run

```sh
# ReAct with the paid Anthropic API (reads ANTHROPIC_API_KEY from the environment)
python -m react_baseline.run --agent react --llm anthropic --task V2_task_01_tomato --seed 0 \
    --max-env-steps 1000 --model claude-haiku-4-5-20251001 --history 10 --actions-per-call 1

# random baseline
python -m react_baseline.run --agent random --task V2_task_01_tomato --seed 0 --max-env-steps 10000

# fully offline (MockLLM returns seeded random well-formed replies)
python -m react_baseline.run --agent react --llm mock --task V2_task_01_tomato --max-env-steps 300
```

| flag | default | meaning |
|---|---|---|
| `--agent` | (required) | `react` or `random` |
| `--task` | `V2_task_01_tomato` | any ID in `external/SAGE/scenarios/overcooked_plus/tasks.json` |
| `--seed` | 0 | seeds Python `random`, NumPy, the action space, the random agent and the Ollama sampler |
| `--seeds` | none | comma-separated list, for example `0,1,2`: one run directory per seed plus a mean ± sd summary; overrides `--seed` |
| `--max-env-steps` | 2000 | budget: env steps |
| `--max-llm-calls` | none | budget: LLM calls |
| `--max-tokens` | none | budget: input + output tokens |
| `--llm` | `ollama` | `ollama`, `anthropic` or `mock` |
| `--model` | per backend | `qwen2.5:7b` / `claude-haiku-4-5-20251001` / `mock` |
| `--temperature` | 0.7 | passed to the backend (0.0 caused a fixed-point loop; see [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)) |
| `--history` | 10 | turns kept in the rolling window |
| `--actions-per-call` | 1 | if > 1, `Action:` may hold up to this many comma-separated actions |
| `--num-ctx` | 8192 | Ollama context window |
| `--ignore-bug-ids` | `5` | oracle IDs dropped from metrics (see the investigation below) |
| `--out-dir` | `runs` | |

The run stops when any budget runs out. Episodes auto-reset when they end and the run
continues. `Ctrl-C` stops the run cleanly and still writes the summary.

## What the agent sees

Each turn the agent gets the system prompt (`prompts.py`) and one user message. The user
message contains the rolling history, the current observation, and a one-line format reminder
if the previous reply could not be parsed. The observation comes from `render_text(env)`:

```
MAP (x = column, y = row; (0,0) is top-left; right = x+1, down = y+1)
    0 1 2 3 4
  0 # # # P #
  1 S @ . . O
  2 T . X . #
  3 K . . . W
  4 # G # L #
Legend: .=floor (walkable), #=empty counter (holds one item), @=you, T=tomato (on a counter), ...
STATIONS
- (0,3) cutting board: empty
- (4,3) sink: empty
- (2,2) delivery window
- (1,4) trash can
ITEMS
- (3,0) plate [clean; contents: empty] (on counter)
- (0,2) tomato [unchopped] (on counter)
...
ORDER: tomato
YOU: at (1,1), holding nothing
ADJACENT (one move away; moving into a non-floor tile interacts with it and you stay in place):
- 0 right -> (2,1): floor (you walk there)
- 1 down  -> (1,2): floor (you walk there)
- 2 left  -> (0,1): steak [unchopped, uncooked] on a counter
- 3 up    -> (1,0): empty counter (holds one item)
STEP: 0/200 in this episode (200 remaining)
LAST EVENT: pickup plate
```

`ADJACENT` lists what occupies each tile one move away, per action ID. It is perception
only: it names the contents and does not predict the outcome of an interaction. It was added
after run #1 showed the model misreading the grid.

`LAST EVENT` is feedback a player would see on screen: moved, picked up, chopped, delivery
accepted or rejected, and so on. It is built from `item_log.event_type` and the success flag.
It contains no bug IDs, and the renderer never reads `info["bugs"]` or the bug manager.
`tests/test_prompt_leak.py` checks this. It patches the bug manager so that some predicates
emit sentinel IDs, then confirms the sentinels reach `steps.jsonl` and never appear in any
prompt.

Each history entry has the form
`[turn k] Thought: … | Action: a | Result: you at (x,y) holding …; event: … [| Bug report: …]`.
An `=== EPISODE RESET ===` marker is inserted when an episode ends. Entries are not summarized.

Reply format: `Thought: …` / `Action: <0-4>[,…]` / optional `Bug report: …`. The parser accepts
case differences, markdown bold, backticks, brackets, a trailing `(left)`, and action names
such as `left`. A reply with no parseable action runs **action 4 (wait)**, logs a
`parse_error`, and adds a format reminder to the next turn. `Bug report: None`-style lines
are not counted as reports.

## Log schema

Each run writes to `runs/<YYYYmmdd-HHMMSS>_<agent>_<task>_seed<seed>/`.

**config.json**: `cli_args` (every CLI argument), `backend`, `model`, `command`, `task`,
`seed`, `agent`, `ignore_bug_ids`, `budgets`, `git_commit` (this repo; `null` until the repo
has a commit), `git_dirty`, `sage_commit`, `started_at`.

**steps.jsonl**, one line per env step:

| field | type | |
|---|---|---|
| `global_step` | int | 1-based, across episodes |
| `episode` | int | 1-based |
| `episode_step` | int | env's step counter after this step (1–200) |
| `action` | int | 0 right, 1 down, 2 left, 3 up, 4 wait |
| `reward` | float | env reward (not shown to the agent) |
| `done` | bool | |
| `bug_ids_triggered` | int[] | oracle `info["bugs"]` minus `ignore_bug_ids`, deduplicated and sorted |
| `new_unique_bug_ids` | int[] | IDs seen for the first time in the run |
| `bug_ids_raw` | int[] | unfiltered `info["bugs"]`, so other filters can be applied later |
| `delivery_success` | bool | an item_log event `deliver` with `others.success == True` |
| `llm_call_id` | int or null | the call that chose this action (null for random) |

**llm_calls.jsonl**, one line per call: `llm_call_id`, `backend`, `model`, `global_step` (steps
done before the call), `episode`, `system_prompt`, `prompt` (the full user message), `response`
(full), `parsed_thought`, `parsed_actions`, `parsed_bug_report`, `input_tokens`,
`output_tokens`, `latency_s`, `parse_error` (null or a reason), `parse_note` (for example,
actions truncated to `--actions-per-call`), `history_turns_used` (lower than `--history` when
the context guard dropped turns).

**bug_reports.jsonl**, one line per agent bug report: `llm_call_id`, `global_step` (the step
whose observation prompted the report), `episode`, `report`, `window` (`[g-3, g+3]`) and
`bug_ids_in_window` (the union of `bug_ids_triggered` over that window). Use these for
precision matching.

**summary.json**: `agent`, `task`, `seed`, `backend`, `model`, `stop_reason`,
`parse_error_rate`, `parse_errors`, `total_env_steps`, `episodes`, `llm_calls`, `input_tokens`,
`output_tokens`, `total_tokens`, `wall_time_s`, `unique_bug_ids`, `n_unique_bugs`,
`first_found_step` (`{bug_id: global_step}`), `discovery_curve` (one point per new discovery
plus a final point, each `{global_step, cumulative_tokens, unique_bugs}`), `deliveries_success`,
`delivery_attempts`, `agent_bug_reports`, `ignore_bug_ids`.

## Investigation: bug IDs 5 and 20 on V2_task_01_tomato

The task brief said that bugs 5 and 20 fired exactly once per episode in random play. I checked
whether they are reset or first-step artifacts. **Neither is**, and in my runs neither fires a
fixed number of times per episode. Over 50 seeded random episodes, bug 5 fired 0–2 times per
episode and bug 20 fired 0–8 times. Neither ever fired on step 1.

- **5 = `BUG_ALL_CUTBOARD_CHOPPED`.** `check_all_cutboards()` runs after every chop that
  completes. It fires when *every* cutting board holds a chopped item. Every shipped map has
  exactly one cutting board, so the condition holds after every completed chop, and that chop
  is a required step of the correct recipe. All five verified demonstrations that include a chop
  trigger bug 5 at the chop step (tomato step 5, lettuce step 7, onion step 9, steak step 6,
  fish step 7). Random play triggers it with the same event (`chop`, success). Bug 5 therefore
  measures normal play on these maps rather than an edge case. **It is in the default
  `ignore_bug_ids`.** It is still recorded in `bug_ids_raw`.
- **20 = `BUG_USE_SINK_WITHOUT_PLATE`.** This fires when the agent bumps the sink with empty
  hands while the sink is empty. On the tomato map that is one specific pointless action (moving
  right into the sink from (3,3)), so random play hits it often. It is a real edge-case
  interaction predicate, not an artifact, and **it is kept**.

Related finding (not filtered): **21 = `BUG_COOK_AFTER_DONE`** fires on every step that a
cooked steak stays in the pan. The verified steak demonstration triggers it 7 times during
correct play. Consider filtering it for steak tasks.

## Env quirks found

- **Coordinates are transposed internally.** `item.x` is the **row** and `item.y` is the
  **column**, and the grid is `map[item.x][item.y]`. Action 0 ("right") does `y += 1` and
  action 1 ("down") does `x += 1`. The YAML `dimensions: [cols, rows]` is unpacked as
  `ylen, xlen`, so `xlen` is the number of rows. The renderer converts everything to screen
  coordinates (x = column, y = row). `test_coordinate_convention` checks this.
- **The env reseeds the global RNG.** On fixed-task maps, `TaskManager.replenish_task()`
  calls `random.seed(42)` on every reset. Any agent that draws from the global `random` module
  would replay the same sequence every episode. `RandomAgent` therefore uses its own
  `random.Random(seed)`, and a test checks that episodes differ. Env dynamics have no other
  randomness, so a run is reproducible from its seed. Two runs with the same seed give
  byte-identical `steps.jsonl` (tested).
- **No shipped map sets `max_trials`**, so the delivery-attempt limit is infinite. In practice
  episodes end only on a correct delivery or at step 200.
- **Every food has `required_chopped_times = 1`**, including rice, cheese and dough, but the
  cutting board only accepts tomato, lettuce, onion, steak and fish. The renderer shows a chop
  state only for those five.
- **The pan is created with `burned_able=False`**, so food never burns in a pan on any shipped
  map. No shipped map has an oven or a block. The renderer handles them anyway.
- After any delivery the plate **respawns dirty** at its start tile (on maps with a sink), and
  the ingredients respawn at their start tiles.
- `info["item_log"]` always has exactly one entry per step for a single agent. Bug IDs are
  removed from its entries before `info` is returned, so passing `event_type` to the agent
  leaks nothing.
- Synthesised dishes (sushi, pizza, roast fish) are pre-allocated with position (-1, -1) and
  `consumed=True` until they are made. The renderer skips them.
- gym 0.26 prints a NumPy-2 deprecation banner on import. The adapter suppresses it.

## Pilot gate

Before any full run, do a 50-step pilot and check it:

```sh
python -m react_baseline.run --agent react --llm ollama --seed 0 --max-env-steps 50 --out-dir runs/pilot
python -m react_baseline.pilot_check runs/pilot/<run_dir>
```

The checker replays `steps.jsonl` through the env, which is deterministic, and requires:
≥ 5 distinct tiles visited, ≥ 1 pickup, a longest same-action streak ≤ 10, a parse error
rate ≤ 20%, and an exact replay match. Exit code 0 means pass. Both qwen2.5:7b attempts so
far failed this gate; see [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

## Measured results and cost

> **The ReAct numbers below come from run #1 (temperature 0, no `ADJACENT` block) and are
> not a valid baseline**: the agent was stuck in a fixed point. They are kept as a record of
> the failure. The random-agent numbers are valid.

These are the definition-of-done runs on `V2_task_01_tomato`, seed 0, with the default
`ignore_bug_ids=[5]`. The ReAct run used local Ollama (`qwen2.5:7b`, temperature 0, history 10,
1 action per call, `num_ctx` 8192). The Anthropic API was never called.

| | random | ReAct (qwen2.5:7b via Ollama) |
|---|---|---|
| env steps | 10,000 | 300 |
| episodes | 51 | 2 |
| unique bug IDs | **17**: 3, 4, 9, 13, 15, 16, 17, 19, 20, 28, 42, 47, 51, 52, 53, 54, 67 | **0** |
| unique bug IDs in the first 300 steps | 8: 15, 16, 19, 20, 28, 42, 52, 53 | 0 |
| LLM calls | 0 | 300 |
| tokens (in / out) | 0 | 547,726 (530,641 / 17,085) |
| successful deliveries / attempts | 1 / 34 | 0 / 0 |
| agent bug reports | – | 299 (1 distinct text, 0 with an oracle bug in its ±3 window) |
| parse error rate | – | 0.0% (0/300) |
| wall time | 1.5 s | 4,130 s (69 min) |

What the ReAct run did: from call 3 onward, qwen2.5:7b at temperature 0 gave the **same reply
every turn**: "pick up the tomato", `Action: 1`, plus a bug report claiming the pickup fails.
The agent stood at (1,3) walking into the trash can for 297 steps. The observation reported
this correctly each time ("LAST EVENT: nothing happened (bumped trash_can)", tomato listed at
(0,2)), and the rolling history showed the repeated failure. A deterministic small model with
history-only memory locked into a fixed point. The prompt was deliberately not tuned to fix
this; this is the baseline behavior.

**Cost per 100 ReAct steps (measured):** about 177k input + 5.7k output tokens (about 1,770
input tokens per call with a full 10-turn history). On this laptop (RTX 3050 4 GB; the model
runs about 50% CPU and 50% GPU) one call has a median latency of 14 s, so 100 steps take about
23 min. Ollama costs **$0**. For reference only: the same token counts at Claude Haiku 4.5
list prices ($1 / MTok input, $5 / MTok output; check current pricing) would cost about
**$0.21 per 100 steps**. The Claude tokenizer differs from Qwen's, so treat that figure as a
rough estimate.

Operational note: during the first attempt the Ollama desktop app auto-updated
(0.5.7 → 0.40.0) and restarted its server at call 41, which killed the run. That partial run
is kept as `runs/*_ABORTED`. The Ollama backend now retries transport errors with backoff
(about 3 min in total) before giving up, and `run()` writes `summary.json` and
`bug_reports.jsonl` with `stop_reason: "error: …"` even when a run crashes. For long runs,
consider turning off Ollama's auto-update.

## Tests

`python -m pytest`. All tests run offline (44 tests, about 7 s).

1. `test_env_adapter.py`: replays all 6 `verified_demonstrations.json` sequences through the
   adapter, rendering at every step, and checks for a successful delivery. Also covers the
   coordinate convention, distinct legend characters, and render snapshots for 3 tasks (to
   regenerate them, set `UPDATE_SNAPSHOTS=1`).
2. `test_prompt_leak.py`: the renderer, summaries, system prompt and every logged prompt
   contain no bug-constant names, no bug-ID sentinels and no oracle markers.
3. `test_agents.py`: parser cases, the wait fallback and format reminder, multi-action calls,
   bug-report capture with the ±3 window, history window and reset marker, the context guard,
   and the LLM-call and token budgets. All of these use MockLLM.
4. `test_runner.py`: a 500-step random run writes every file with the documented schema, runs
   with the same seed are identical, and episodes differ even though the env reseeds the
   global RNG.

## Attribution

Overcooked Plus is from SAGE (https://github.com/BlueLinkX/SAGE), which builds on
https://github.com/BlueLinkX/Overcooked-Plus and https://github.com/rosewang2008/gym-cooking.
It has no license, so it is not redistributed here.
