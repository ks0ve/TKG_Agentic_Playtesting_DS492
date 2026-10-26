# TKG Agentic Playtesting: ReAct baseline

The baseline arm of a study that compares LLM-agent exploration with algorithmic baselines
for game bug discovery. The game is **Overcooked Plus** from
[SAGE](https://github.com/BlueLinkX/SAGE). The package `react_baseline/` contains:

- a **shared Action reader** (`action_reader.py`): a deterministic, LLM-free action interface
  that turns intents such as `goto(3,0)`, `interact(0,3)`, `wait(5)` or a raw move into button
  presses 0–4. **Every agent goes through it**, so the only difference between experimental arms
  is *when* the LLM is consulted, not how coarse the action interface is
  (see [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md), 2026-10-20).
- a **vanilla ReAct agent** (Thought → Action → Observation) in two action spaces: `react`
  (intents, the headline baseline) and `react-raw` (raw moves 0–4, the MVP, kept as an
  ablation). Its only memory is a rolling window of the last N turns. It has no planner,
  reflection, skill library or graph memory. The Action reader is the shared action interface,
  not a planner.
- two no-LLM baselines through the same harness and logging: `random` (uniform over the 5 raw
  actions) and `random-intent` (uniform over the currently valid intents).
- a harness with budgets (env steps, LLM calls, tokens, USD) and auto-reset. It also keeps a
  **hidden oracle**: the env's bug-predicate IDs are logged but never shown to the agent.
- a **model registry** (`configs/models.yaml`): swapping a local or self-hosted model is one
  YAML entry.

```
react_baseline/
  env_adapter.py   make_env, render_text, visible_state, EnvAdapter.step -> (agent-visible, hidden oracle)
  action_reader.py intent parser + executor (BFS paths, stop rules), valid_intents
  transitions.py   VisibleTransition (one per executed intent -> transitions.jsonl)
  prompts.py       system prompt + per-turn template (primitive and intent action sections)
  llm.py           model registry + Ollama / OpenAI-compatible / Anthropic / Mock backends
  agents.py        ReActAgent (intent | primitive), RandomAgent, RandomIntentAgent, parsers
  runner.py        budgeted loop through the Action reader, auto-reset, all log files, attribution
  taxonomy.py      draft bug-taxonomy counts for summary.json (reporting only)
  run.py           CLI
  pilot_check.py   pass/fail gate for a short pilot run
  tests/           pytest (offline)
configs/
  models.yaml              model profiles
  bug_taxonomy_draft.csv   draft classification of all SAGE bug IDs (status=proposed)
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
PyYAML 6.0, networkx 3.4.2) and adds `anthropic`, `ollama`, `httpx` (already a dependency of
`ollama`; used by the OpenAI-compatible backend) and `pytest`. The adapter adds
`external/SAGE/scenarios/overcooked_plus` to `sys.path`. It also sets `SDL_VIDEODRIVER`,
`SDL_AUDIODRIVER` and `PYGAME_HIDE_SUPPORT_PROMPT`, and silences the gym/NumPy-2 deprecation
banner.

## Running for free (Ollama, the default)

The default LLM backend is a local Ollama server. Nothing calls a paid API unless you select an
Anthropic profile (`--model-profile haiku-5.5` or `--llm anthropic`) together with `--allow-paid`.

1. Install Ollama from https://ollama.com/download.
2. Start the server if it is not already running: `ollama serve`
3. Pull the default model (4.7 GB): `ollama pull qwen2.5:7b`
4. Pilot first (50 steps, checked automatically), then run:

```sh
python -m react_baseline.run --agent react --model-profile qwen2.5-7b --pilot
python -m react_baseline.run --agent react --model-profile qwen2.5-7b --task V2_task_01_tomato --max-env-steps 300
```

If the server is unreachable or the model has not been downloaded, the CLI exits with code 2.
The message names the exact command to run (`ollama serve` or `ollama pull <model>`).

The Ollama call is `ollama.chat(model, messages, options={temperature, num_ctx, seed,
num_predict})`. `num_ctx` is the profile's `context_window` (8192 for the shipped profiles) and
`num_predict` its `max_output_tokens` (512). Token accounting uses the response's
`prompt_eval_count` (input) and `eval_count` (output), so the logs have the same schema for
every backend.

**Context-overflow guard.** Before each call the agent estimates the prompt size as
characters / 3.5. If the estimate is above 90% of the profile's context window, it prints a
`[context guard]` warning and drops the oldest history turns until the prompt fits. If even a
prompt with no history does not fit, the run stops with an error asking for a larger
`--num-ctx`. The system prompt is never cut, so the server never has to truncate it silently.

## Model registry (`configs/models.yaml`)

Each named profile has `backend` (`ollama` | `openai_compat` | `anthropic` | `mock`), `model`,
`base_url`, `context_window`, `max_output_tokens`, default `temperature`, `price_in_per_mtok` /
`price_out_per_mtok` (USD; 0 for local models) and backend `options`. **Adding a model is one
YAML entry, no code change.** Select it with `--model-profile <name>`; `--llm` / `--model` /
`--temperature` / `--num-ctx` / `--max-output-tokens` / `--ollama-host` still work and override
the profile. Without `--model-profile`, the `--llm` backend's default profile is used
(`ollama` → `qwen2.5-7b`, `anthropic` → `haiku-5.5`). If `--model` names an id that matches no
profile, the price is unknown: `usd_cost` is `null` and `--max-usd` is refused.

| profile | backend | model | $/MTok in / out |
|---|---|---|---|
| `qwen2.5-7b` (default) | ollama | `qwen2.5:7b` | 0 / 0 |
| `qwen2.5-14b` | ollama | `qwen2.5:14b` | 0 / 0 |
| `llama3.1-8b` (replication) | ollama | `llama3.1:8b` | 0 / 0 |
| `mistral-nemo-12b` | ollama | `mistral-nemo:12b` | 0 / 0 |
| `vllm-qwen2.5-7b` | openai_compat | `Qwen/Qwen2.5-7B-Instruct` at `http://localhost:8000` | 0 / 0 |
| `haiku-4.5` (**disabled**) | anthropic | `claude-haiku-4-5-20251001` | 1.00 / 5.00 |
| `haiku-5.5` (headline) | anthropic | `claude-haiku-5-5` | 0.10 / 0.50 (prompts ≤ 100k tokens) |
| `mock` | mock | offline | 0 / 0 |

Anthropic prices were checked on https://platform.claude.com/docs/en/about-claude/pricing on
2026-10-20; each profile records `pricing_source` and `pricing_date`. Two API details matter:

- `anthropic` 1.11.0's `messages.create` has **no `temperature` keyword**, so temperature is sent
  in `extra_body`. (The MVP passed it as a keyword, which raises `TypeError`; the paid path had
  never been exercised.)
- **Claude Haiku 5.5 rejects non-default sampling values** (HTTP 400), and adaptive thinking is
  on by default. The `haiku-5.5` profile therefore sends no temperature (`send_temperature:
  false`) and disables thinking (`extra_body: {thinking: {type: disabled}}`), so it runs at the
  API's default temperature, unlike the 0.7 used by the local profiles. `haiku-4.5` (now disabled) honours
  temperature 0.7 and is the like-for-like comparison with local runs.

**OpenAI-compatible servers** (vLLM, llama.cpp `llama-server`, LM Studio, TGI): backend
`openai_compat`, `base_url` with or without `/v1`. The backend checks readiness with
`GET /v1/models` (and that the model id is served), posts to `/v1/chat/completions`, reads
tokens from `usage`, passes `seed` unless `options.supports_seed: false`, reads a bearer token
from the env var named in `options.api_key_env`, and retries transport errors and 429/502/503/504
with the same backoff as Ollama (about 3 min in total).

## Other ways to run

```sh
# ReAct with the paid Anthropic API (reads ANTHROPIC_API_KEY), capped at $1
python -m react_baseline.run --agent react --model-profile haiku-5.5 --seed 0 --max-env-steps 300 --allow-paid --max-usd 1

# the MVP ReAct (raw moves) as an ablation
python -m react_baseline.run --agent react-raw --model-profile qwen2.5-7b --max-env-steps 300

# no-LLM baselines
python -m react_baseline.run --agent random --seed 0 --max-env-steps 10000
python -m react_baseline.run --agent random-intent --seed 0 --max-env-steps 10000

# fully offline (the mock returns seeded random well-formed replies / intents)
python -m react_baseline.run --agent react --llm mock --max-env-steps 300
```

| flag | default | meaning |
|---|---|---|
| `--agent` | (required) | `react` (intents), `react-raw` (raw moves), `random`, `random-intent` |
| `--task` | `V2_task_01_tomato` | any ID in `external/SAGE/scenarios/overcooked_plus/tasks.json` |
| `--seed` | 0 | seeds Python `random`, NumPy, the action space, the random agents and the LLM sampler (Ollama / openai_compat) |
| `--seeds` | none | comma-separated list, for example `0,1,2`: one run directory per seed plus a mean ± sd summary; overrides `--seed` |
| `--max-env-steps` | 2000 | budget: env steps (**the primary comparison axis**) |
| `--max-llm-calls` | none | budget: LLM calls |
| `--max-tokens` | none | budget: input + output tokens |
| `--max-usd` | none | budget: USD at the profile's prices; checked before each call, so a run can overshoot by at most one call |
| `--model-profile` | none | profile in `configs/models.yaml` |
| `--llm` | `ollama` | backend; selects that backend's default profile |
| `--model` | profile's | override the model id |
| `--temperature` | profile's (0.7) | 0.0 caused a fixed-point loop; see [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) |
| `--max-output-tokens` | profile's (512) | |
| `--num-ctx` | profile's (8192) | context window (and Ollama `num_ctx`) |
| `--ollama-host` | profile's | override `base_url` |
| `--history` | 10 | turns kept in the rolling window |
| `--intents-per-call` | 1 | `react`: intents per call (1 = consulted at every decision point) |
| `--actions-per-call` | 1 | `react-raw`: raw actions per call |
| `--max-macro-len` | 40 | Action reader: max env steps one intent may take |
| `--ignore-bug-ids` | `5` | oracle IDs dropped from metrics (see the investigation below) |
| `--out-dir` | `runs` | |
| `--pilot` | off | 50 steps into `runs/pilot`, then `pilot_check` on each run; exit code 1 on a fail |
| `--run-dir` | none | exact run directory, single seed (used by the grid driver) |
| `--allow-paid` | off | required for any profile with a price > 0 |
| `--max-usd-override` | off | allow `--max-usd` above the $2 per-run ceiling |
| `--ledger`, `--budget-file` | `spend/ledger.jsonl`, `configs/budget.yaml` | spend guardrail files |
| `--repro` | off | agent-driven reproduction after new bug reports (on in the grid) |
| `--repro-steps` | 100 | env steps per reproduction attempt |
| `--max-repro-attempts` | 5 | per run |
| `--max-repro-fraction` | 0.3 | max share of `--max-env-steps` spent on attempts |
| `--repro-dedup-jaccard` | 0.8 | skip near-duplicate reports (token Jaccard) |

The run stops when any budget runs out. Episodes auto-reset when they end and the run
continues. `Ctrl-C` stops the run cleanly and still writes the summary.

## Action reader (shared by every agent)

The Action reader (`action_reader.py`) is deterministic and calls no LLM. Every agent's output
goes through it; raw moves become one-step intents. Coordinates are in the screen frame the
agent sees (x = column, y = row).

| intent | accepted spellings | executes |
|---|---|---|
| move | `right` `down` `left` `up` `0` `1` `2` `3` | one primitive |
| wait | `wait`, `4`, `stay`, `wait(n)`, `wait n` (1 ≤ n ≤ 20) | n × action 4 (pans and rice cookers cook over time) |
| goto | `goto(x,y)`, `go to (x, y)`, `goto x,y`, `move_to(x,y)` | BFS over floor tiles to (x,y), which must be floor |
| interact | `interact(x,y)`, `use(x,y)`, `interact with x,y`, `interact(sink)` | BFS to the nearest floor tile next to (x,y), then bump into it once; a station/item name must resolve to exactly one tile |

Parsing tolerates case, markdown, backticks, brackets and trailing text. Several intents in one
`Action:` line are separated by `;`, a newline or `then`, and are capped at `--intents-per-call`.

- **Paths:** breadth-first over floor tiles (`.` and the agent's own tile), expanding neighbours
  right, down, left, up (= action IDs 0–3); the first shortest path found wins. The reader plans
  on `visible_state` in screen coordinates; the env's transposed internals (`item.x` = row) are
  handled once in `env_adapter` and tested by `test_goto_every_reachable_tile_lands_in_screen_frame`.
- **Stops early** on episode end (`episode_end`), an exhausted budget (`budget`), a goto step whose
  event is not a plain move to the expected tile (`unexpected_event`), or the length cap
  (`length_cap`, 40 steps). Otherwise `completed`.
- **Floor / non-floor tolerance:** `goto(x,y)` on a non-floor tile walks next to it and stops
  without bumping; `interact(x,y)` on a floor tile walks onto it. Both are noted in the trace and
  logged as `coerced`, not as failures (EXPERIMENT_LOG, 2026-10-20).
- **Invalid intents** (unparseable, outside the map, unreachable, ambiguous or unknown name,
  `wait(25)`) run **one wait** so budgets always advance, and return an agent-visible error such
  as `goto(9,9) failed: outside the map`. A goto to the current tile waits one step with the note
  `already there`.
- **Returns** the final observation plus a compact agent-visible trace, e.g.
  `interact(0,2): 2 step(s) [down,left]; events: moved; pickup tomato; stop: completed`.
- `valid_intents(state)` lists the four moves, `wait`, `goto` to every other reachable floor tile
  and `interact` with every non-floor tile next to a reachable floor tile (used by `random-intent`).

The module holds no agent logic, so the future TKG arm's cheap policy and its LLM reuse it as-is.
Every verified demonstration task can be completed by a short hand-written intent sequence
(`tests/test_action_reader.py::SOLUTIONS`).

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
`[turn k | ep e | steps a–b (n steps)] Thought: … | Action: a | Result: you at (x,y) holding …; event: … [| Bug report: …]`,
where `a–b` are episode steps numbered like the observation's `STEP: k/200` line (one step: `step a (1 step)`).
An `=== EPISODE RESET: the kitchen was reset to its starting state ===` marker is inserted when an
episode ends (reproduction attempts use the same suffix). Entries are not summarized. Event text
describes the outcome from the player's view, using the legend's names, e.g. "put tomato on the
cutting board at (0,3)" or "tried to put tomato down at (1,4), where the onion is: nothing changed".

Reply format: `Thought: …` / `Action: <0-4>[,…]` / optional `Bug report: …`. The parser accepts
case differences, markdown bold, backticks, brackets, a trailing `(left)`, and action names
such as `left`. A reply with no parseable action runs **action 4 (wait)**, logs a
`parse_error`, and adds a format reminder to the next turn. `Bug report: None`-style lines
are not counted as reports.

`react-raw` uses exactly the MVP system prompt; its observations and history come from the shared
translator (snapshot `tests/snapshots/mvp_prompts.json`, regenerated for the v2.1 wording). `react` changes only the actions bullet of the system prompt
(it describes the intent grammar) and the `Action:` spec in the output format and the format
reminder. Its history entries add the executor trace:
`[turn k | ep e | steps a–b (n steps)] Thought: … | Action: interact(0,3) | Executor: interact(0,3): 3 step(s) [down,down,left]; events: moved x2; put tomato on the cutting board at (0,3); stop: completed | Result: … [| Bug report: …]`.
A reply with no `Action:` line or an unparseable first intent runs `wait` and counts as a
parse error; a parseable intent the executor cannot carry out counts as an intent failure.

The agent never sees `visible_state` directly, but it is agent-visible by construction: it is
built from the same game state as `render_text` (agent position and held item, items, stations
with progress, order, map, last event type + target + success) and is covered by the leak test.

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

**v2 additions** (fields are only ever added; nothing above was renamed or removed):

- **config.json**: `action_space` (`intent` | `primitive`), `max_macro_len`, `model_profile`
  (the full resolved profile, prices included; `null` for no-LLM agents).
- **steps.jsonl**: `decision_id` (0-based, one per `agent.act`), `intent_id` (0-based, one per
  executed intent), `intent` (canonical text, e.g. `goto(3,1)`; raw moves appear as `right` …
  `wait`), `intent_step_index` (0-based position of this step inside its intent).
- **llm_calls.jsonl**: `decision_id`, `parsed_intents`, `executor_result` (one
  `{intent, kind, n_steps, stop_reason, error, note}` per executed intent). In intent mode
  `parsed_actions` is `[]`; the executed actions are in steps.jsonl. The row is written after
  the call's intents have run.
- **transitions.jsonl** (new): one `VisibleTransition` per executed intent: `prev_state`,
  `intent`, `primitive_actions`, `events`, `new_state` (agent-visible state dicts), plus
  `decision_id`, `intent_id`, `episode`, `global_step_start`, `global_step_end`, `kind`,
  `event_records`, `stop_reason`, `error`, `note`, `episode_ended`.
- **bug_reports.jsonl**: `decision_id`; **strict attribution**: a report made at decision d is
  about the observation produced by decision d−1, so `bug_ids_strict` = oracle IDs that fired
  during the env steps executed by decision d−1 and `evidence_steps` = those global steps. The
  legacy ±3 `window` / `bug_ids_in_window` (lenient) is unchanged.
- **summary.json**: `action_space`, `usd_cost` (profile price × tokens; 0 for local models),
  `decisions`, `llm_calls_per_decision`, `llm_calls_per_env_step`, `llm_latency_mean_s`,
  `llm_latency_median_s`, `intents_executed`, `intent_failures`, `intent_failure_rate`,
  `reports_strict_match` / `reports_lenient_match` (reports with ≥ 1 strict / window match),
  `unique_bugs_reported_strict` (+ `unique_bug_ids_reported_strict`), and the draft-taxonomy
  counts `bugs_by_proposed_class`, `bugs_by_family`, `taxonomy_warnings`.

**Draft bug taxonomy (measured alongside, never used to filter).**
`configs/bug_taxonomy_draft.csv` classifies every SAGE bug ID (family, reachability, trigger
signature, proposed RQ3 class; all `status=proposed`). `summary.json` counts the run's unique
bug IDs (after `ignore_bug_ids`) by class and family. It changes no metric, filter or behaviour
(tested). If a run finds a bug the CSV marks `unreachable_on_shipped_maps`, the CLI prints a
warning and `taxonomy_warnings` records it. The CSV's classifications are not edited here.

## Paid runs: budget and guardrails

- **Project budget** `configs/budget.yaml`: `project_cap_usd: 20.00`, `safety_margin_usd: 1.00`.
  A paid run starts only if finalized spend + open reservations + its `--max-usd` ≤ $19.00.
- **Ledger** `spend/ledger.jsonl` (tracked in git, append-only): a `reservation` record at run
  start and a `final` record (actual `usd_cost`, tokens, status) at run end, including after
  Ctrl-C or a crash. A reservation without a final record (killed process) counts at its full
  `max_usd` until `python -m react_baseline.spend --reconcile <run_dir>`.
- `python -m react_baseline.spend` prints spent, remaining, per-profile / per-run costs and open
  reservations.
- **Paid profiles** (any price > 0) need `--allow-paid` and `--max-usd` (≤ $2 unless
  `--max-usd-override`). Before the first call the CLI prints an estimate: steps × calls/step ×
  tokens/call × price, from the latest pilot of that agent + model, else labelled conservative
  defaults (1 call/step, 3,000 in + 200 out tokens).
- `haiku-4.5` is `disabled: true` (10× the price); only editing `models.yaml` re-enables it.
- **Haiku 5.5** (`claude-haiku-5-5`, $0.10 / $0.50 per MTok for prompts ≤ 100k tokens; verified
  2026-10-21): thinking disabled (it bills as output), no temperature sent (the API rejects
  sampling parameters), `max_output_tokens: 384`, and a hard abort if a prompt is estimated above
  90k tokens (above 100k the whole request is billed 5×).
- **Errors**: SDK retries are off; 429 / 5xx / connection errors are retried with backoff; any
  other API error (e.g. a 400 validation error) stops the run with a clear message; 5 consecutive
  API errors stop the run. A response without `usage` is refused.
- **No prompt caching.** Haiku 5.5's minimum cacheable prompt is 512 tokens and our ~600-token
  system prompt would qualify, but caching it would save fractions of a cent per call, so it is
  left off deliberately (one fewer moving part in the cost accounting).

## Agent-driven reproduction (`--repro`)

A runner-level protocol with optional agent hooks (`on_repro_start`, `on_repro_tick`,
`on_repro_end`), so the future TKG arm reuses it unchanged.

1. A bug report filed while exploring starts an attempt unless it is a near-duplicate of an
   already-attempted report (same normalized text or token Jaccard ≥ 0.8), 5 attempts were made,
   or the attempts would exceed 30% of the step budget. This is decided from agent-side
   information only; the oracle is never consulted. With the defaults (100-step attempts, 30% cap) a run needs
   ≥ 334 steps for any attempt; short pilots should pass e.g. `--repro-steps 15`.
2. The env resets (`episodes.jsonl` end_reason `repro_start`), the history gets
   `=== REPRODUCTION ATTEMPT ===`, and each turn prompt carries
   `REPRODUCTION TASK: reproduce the defect you reported: "<report>". Steps left: k. If you believe
   it happened again, add a line "Reproduced: yes".`
3. The attempt ends after 100 env steps, a `Reproduced: yes` reply, or the run budget. Reports
   during an attempt are logged but never start another. Then the env resets, the history gets
   `=== END REPRODUCTION ===`, and exploration resumes.
4. Scoring (oracle, post hoc): `oracle_reproduced` = any of the source report's `bug_ids_strict`
   fire during the attempt (`null` if the report had no strict match), plus the agent's claim,
   `first_repro_step`, `other_bug_ids_seen`, and steps / calls / tokens / $.

Logs: `repro_attempts.jsonl` (one row per attempt), `episodes.jsonl` (one row per episode with its
end reason), and `phase` (`explore` | `repro`) + `repro_attempt_id` on steps, llm_calls,
transitions and bug_reports. Summary: `{steps,llm_calls,tokens,usd}_{explore,repro}`,
`n_unique_bugs_explore` (next to the unchanged `n_unique_bugs`), `repro_attempts`,
`repro_skipped`, `repro_success_rate` (over attempts with a strict-matched source),
`unique_bugs_reproduced`, `agent_claim_accuracy`. Repro steps count against the run's budgets.

## Post-hoc metrics (reporting only)

- `loop_decision_fraction`: share of decisions inside runs of ≥ 3 identical decisions;
  `cycle2_decision_fraction`: share inside A,B,A,B alternations of length ≥ 4 (the 14b's tomato
  juggling and the 7b's goto/interact loop show up here).
- **Scripted correct play**: `configs/solutions/<task>.yaml` completes each grid task through the
  Action reader (tested).
- **Normal-play bug sets**: `configs/normal_play_bugs.yaml` lists the oracle IDs that fire during
  correct play. For the grid tasks this is the union over every correct-play variant: preparation
  order, assembly order, plating method (scoop into the held plate from a station or counter, or
  put down onto the plate) and putting the plate down partway. It comes from
  `python -m react_baseline.normal_play --grid --write`, with a per-variant breakdown. Other tasks
  keep the scripted solution's IDs (`python -m react_baseline.solutions --write`). Used for summary
  `unique_bugs_excluding_normal_play` and the `*_adj` metrics. `ignore_bug_ids` stays `[5]`.
- **Adjusted metrics**: `python -m react_baseline.rescore <run_dir|grid_dir>` writes `summary_rescored.json` (the summary plus
  `*_adj` fields with the normal-play IDs removed: reports, reproduction, taxonomy classes) and prints an
  arm x task table. `summary.json` is never modified.

## Grid, frozen config and bundles

- **Grid** `configs/grids/baseline_v2.yaml`: tasks 01 tomato, 07 tomato + steak, 11 tomato +
  lettuce + onion; 1,000 steps; history 10; temperature 0.7 where supported; $1 per paid run;
  repro on. Rows: `react` × haiku-5.5 (5 seeds), `react` and `react-raw` × llama3.1-8b (3 seeds),
  `random` and `random-intent` (3 seeds).
- `python -m react_baseline.grid configs/grids/baseline_v2.yaml [--only react/llama3.1-8b]
  [--dry-run] [--allow-paid] [--allow-unfrozen]`. Sequential and resumable: a combination is done
  when `runs/grids/baseline_v2/<agent>_<profile>_<task>_seed<N>/summary.json` has stop_reason
  `max_env_steps` and the same config hash; anything else is re-run (the old dir is kept as
  `*_stale<timestamp>`). Paid combinations go through every guardrail above. `--dry-run` shows
  status and estimated wall time / $ per combination.
- **Frozen config** `configs/frozen/baseline_v2.yaml` records everything that defines the
  baseline (prompt hash, Action-reader version, agent / repro / budget settings, tasks, profiles,
  `ignore_bug_ids`). The grid refuses to run if the code's prompt hash or reader version no longer
  match (`python -m react_baseline.frozen --check configs/frozen/baseline_v2.yaml`).
- **Bundle** `python -m react_baseline.bundle <run_dir> ... -o out.zip`: per-run logs plus a fresh
  `pilot_check.txt`, the ledger, the spend report, git HEAD / status and the grid dry run, in one
  zip for analysis elsewhere.

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

Before any full run, do a 50-step pilot. `--pilot` runs it into `runs/pilot` and checks it:

```sh
python -m react_baseline.run --agent react --model-profile qwen2.5-7b --seed 0 --pilot
# or check any run dir by hand:
python -m react_baseline.pilot_check runs/pilot/<run_dir>
```

The checker replays `steps.jsonl` through the env, which is deterministic. Every agent needs
≥ 5 distinct tiles visited, ≥ 1 pickup, a parse error rate ≤ 20% and an exact replay match.
Primitive-action agents (`react-raw`, `random`) also need a longest same-action streak ≤ 10.
Intent agents (`react`, `random-intent`) instead need a longest run of identical intents ≤ 10
and an intent-failure rate ≤ 30%: a `goto` legitimately repeats one raw move, so the raw streak
is not a loop signal there. Exit code 0 means pass. See [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)
for the pilot history.

## Measured results and cost

> **This section records the MVP (pre-Action-reader) runs.** v2 mock verification numbers and
> the qwen2.5:7b pilots in both action spaces are in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)
> (2026-10-20). The random-agent numbers below are unchanged in v2: the same seed reproduces
> the same 17 bug IDs in 10,000 steps.

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

`python -m pytest`. All tests run offline and never call an LLM (paid or local); about 40 s.
Since 2026-10-21 also: `test_spend.py` (ledger, cap, reservations, paid-profile rules, Anthropic
error policy with a fake client), `test_repro.py`, `test_metrics.py`, `test_solutions.py`,
`test_grid_frozen.py`, `test_bundle.py`.

1. `test_env_adapter.py`: replays all 6 `verified_demonstrations.json` sequences through the
   adapter, rendering at every step, and checks for a successful delivery. Also covers the
   coordinate convention, distinct legend characters, and render snapshots for 3 tasks (to
   regenerate them, set `UPDATE_SNAPSHOTS=1`).
2. `test_prompt_leak.py`: the renderer, summaries, system prompts (both action spaces), every
   logged prompt, the structured state, executor feedback (traces, errors, `executor_result`)
   and `transitions.jsonl` contain no bug-constant names, no bug-ID sentinels and no oracle
   markers, while the sentinels do reach `steps.jsonl`.
3. `test_agents.py`: parser cases, the wait fallback and format reminder, multi-action calls,
   bug-report capture with the ±3 window, history window and reset marker, the context guard,
   and the LLM-call and token budgets (MockLLM).
4. `test_runner.py`: a 500-step random run writes every file with the documented schema, runs
   with the same seed are identical, and episodes differ even though the env reseeds the
   global RNG.
5. `test_action_reader.py`: intent parsing; BFS tie-break and paths on the snapshot maps;
   `goto` to every reachable tile on 3 maps lands at screen (x,y) = env (item.y, item.x);
   interact; name resolution; invalid intents run one wait with an error; the no-op goto; the
   length cap; budget and unexpected-bump stops; episode end; a hand-written intent sequence
   that completes each verified-demonstration task; transitions.jsonl consistency.
6. `test_agents_v2.py`: `react-raw` prompts match the MVP snapshot byte-for-byte; intent mode
   changes only the action section; intent-response parsing; intent ReAct logs, history and
   fallbacks; determinism of mock `react` and `random-intent`; the CLI agents and `--pilot`;
   the intent pilot gate.
7. `test_llm_registry.py`: models.yaml parsing (a new model is one entry); overrides and unknown
   prices; the `openai_compat` backend against a mocked httpx transport (usage, seed, retries,
   setup errors); Anthropic requests bind to the installed SDK's signature; `--max-usd` stops a
   run; local profiles cost $0.
8. `test_attribution_taxonomy.py`: strict vs lenient report matching on synthetic step rows and
   in a run; taxonomy counts and warnings; the taxonomy changes no metric.

## Attribution

Overcooked Plus is from SAGE (https://github.com/BlueLinkX/SAGE), which builds on
https://github.com/BlueLinkX/Overcooked-Plus and https://github.com/rosewang2008/gym-cooking.
It has no license, so it is not redistributed here.
