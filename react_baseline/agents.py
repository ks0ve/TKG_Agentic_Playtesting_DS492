"""Agents sharing one interface: ``act(obs_text, state) -> Decision``.

``Decision.actions`` holds 1..actions_per_call raw env actions to run in order,
or ``Decision.intents`` holds intents for the shared Action reader. Either way
the runner executes them through the reader. ``Decision.llm_call`` is the record
for llm_calls.jsonl (None for agents without an LLM). After execution the runner
calls ``observe(executed, summary, results)`` (results = one ExecResult per
intent), and ``on_episode_reset()`` when the env resets.

Agents: ``ReActAgent(action_space="intent")`` (CLI ``react``, the headline
baseline), ``ReActAgent(action_space="primitive")`` (CLI ``react-raw``, the MVP
ablation; MVP system prompt except the v2.1 Bug-report format sentence, observations and
history from the shared translator), ``RandomAgent`` (uniform over the
5 raw actions) and ``RandomIntentAgent`` (uniform over currently valid intents).
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from . import prompts
from .action_reader import Intent, parse_intents, valid_intents
from .repro import END_MARKER, RESET_NOTE, START_MARKER, parse_reproduced, task_block
from .llm import estimate_tokens

WAIT = 4
_NAME_TO_ACTION = {"right": 0, "down": 1, "left": 2, "up": 3, "wait": 4, "stay": 4}


@dataclass
class Decision:
    actions: list[int] = field(default_factory=list)
    llm_call: dict | None = None
    intents: list | None = None     # intent strings/Intents; if None, `actions` run as primitives
    results: list | None = None     # filled by the runner: one ExecResult per executed intent


@dataclass
class ParsedResponse:
    thought: str = ""
    actions: list[int] = field(default_factory=list)
    bug_report: str | None = None
    parse_error: str | None = None
    parse_note: str | None = None
    intents: list = field(default_factory=list)   # intent mode only
    reproduced: bool = False                      # reply had 'Reproduced: yes'
    bug_report_discarded: str | None = None       # a 'Bug report:' line that is not a report


_ACTION_LINE = re.compile(r"^[\s*#>_`-]*action[\s*_`]*(?:\(s\))?[\s*_`]*[:=：]\s*(.*)$",
                          re.IGNORECASE | re.MULTILINE)
_ACTION_ANYWHERE = re.compile(r"action[\s*_`]*[:=：]\s*([^\n]*)", re.IGNORECASE)
_THOUGHT = re.compile(
    r"^[\s*#>_`-]*thought[\s*_`]*[:：]\s*(.*?)(?=^[\s*#>_`-]*(?:action|bug[ _]?report)\b|\Z)",
    re.IGNORECASE | re.MULTILINE | re.DOTALL)
_BUG = re.compile(r"^[\s*#>_`-]*bug[ _]?report[\s*_`]*[:：]\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_NUM_LIST = re.compile(r"^([0-4](?:\s*[,;]?\s+[0-4]|\s*[,;]\s*[0-4])*)(?![0-9])")
_NAME_LIST = re.compile(r"^((?:right|down|left|up|wait|stay)(?:\s*[,;]\s*(?:right|down|left|up|wait|stay))*)\b",
                        re.IGNORECASE)
_NONE_REPORT = {"", "none", "n/a", "na", "no", "nothing", "-", "null", "none."}
# "Bug report:" lines that say there is no (confirmed) defect: logged, never counted as reports
_NON_REPORT_START = ("unconfirmed", "no defect", "no bug", "nothing", "not a bug")
_NON_REPORT_ANY = re.compile(r"\bno (?:clear )?(?:defect|bug) (?:is )?(?:confirmed|found|observed)\b",
                             re.IGNORECASE)


def is_non_report(report: str) -> bool:
    low = report.lower().lstrip("*_`'\" ")
    return low.startswith(_NON_REPORT_START) or bool(_NON_REPORT_ANY.search(report))


def parse_response(text: str, actions_per_call: int = 1) -> ParsedResponse:
    """Tolerant parse of 'Thought: ... / Action: ... / [Bug report: ...]'."""
    out = ParsedResponse()
    out.reproduced = parse_reproduced(text)
    m = _THOUGHT.search(text or "")
    if m:
        out.thought = " ".join(m.group(1).split())
    b = _BUG.search(text or "")
    if b:
        report = b.group(1).strip().strip("*_` ").strip()
        if report.lower() in _NONE_REPORT or report.lower().startswith("none"):
            pass
        elif is_non_report(report):
            out.bug_report_discarded = report
        else:
            out.bug_report = report

    a = _ACTION_LINE.search(text or "") or _ACTION_ANYWHERE.search(text or "")
    if not a:
        out.parse_error = "no 'Action:' line found"
        return out
    raw = a.group(1).strip().strip("*_`'\"[]() ").strip()
    nums = _NUM_LIST.match(raw)
    if nums:
        actions = [int(x) for x in re.findall(r"[0-4]", nums.group(1))]
    else:
        names = _NAME_LIST.match(raw)
        if not names:
            out.parse_error = f"unparseable action {raw[:60]!r}"
            return out
        actions = [_NAME_TO_ACTION[n.lower()] for n in re.findall(r"[a-zA-Z]+", names.group(1))]
    if len(actions) > actions_per_call:
        out.parse_note = f"truncated {len(actions)} actions to {actions_per_call}"
        actions = actions[:actions_per_call]
    out.actions = actions
    return out


def parse_intent_response(text: str, intents_per_call: int = 1) -> ParsedResponse:
    """Like ``parse_response``, but the Action line holds intents (Action reader
    grammar). No Action line, or a first intent that cannot be parsed, is a parse
    error (the agent then waits, and gets the format reminder)."""
    out = parse_response(text, 99)
    out.actions, out.parse_note = [], None
    a = _ACTION_LINE.search(text or "") or _ACTION_ANYWHERE.search(text or "")
    if not a:
        out.parse_error = "no 'Action:' line found"
        return out
    out.parse_error = None
    intents, note = parse_intents(a.group(1), intents_per_call)
    if intents[0].kind == "invalid":
        out.parse_error = intents[0].error
        return out
    out.intents = [i for i in intents if i.kind != "invalid"]
    dropped = len(intents) - len(out.intents)
    notes = [note, f"dropped {dropped} unparseable intent(s)" if dropped else None]
    out.parse_note = "; ".join(x for x in notes if x) or None
    return out


class RandomAgent:
    name = "random"
    action_space = "primitive"

    def __init__(self, seed: int = 0):
        # A private RNG: the env calls random.seed(42) on every reset (fixed-task
        # maps), so the global `random` module would replay the same sequence.
        self.rng = random.Random(seed)

    def act(self, obs_text: str, state: dict | None = None) -> Decision:
        return Decision([self.rng.randrange(5)])

    def observe(self, executed: list[int], summary: str, results=None):
        pass

    def on_episode_reset(self):
        pass


class RandomIntentAgent(RandomAgent):
    """No-LLM baseline through the same Action reader: one intent per decision,
    uniform over ``valid_intents(state)`` (moves, wait, goto every reachable
    floor tile, interact every reachable non-floor tile)."""

    name = "random-intent"
    action_space = "intent"

    def act(self, obs_text: str, state: dict | None = None) -> Decision:
        return Decision(intents=[self.rng.choice(valid_intents(state))])


class ContextOverflowError(RuntimeError):
    pass


class ReActAgent:
    """Vanilla ReAct: Thought -> Action -> Observation, with only a rolling
    window of the last ``history`` turns as memory.

    ``action_space="primitive"``: Action is 1..actions_per_call raw moves (MVP).
    ``action_space="intent"``: Action is 1..intents_per_call intents for the
    Action reader; the history records the intent, the executor trace and the
    result. The agent is consulted at every decision point either way."""

    RESET_MARKER = f"=== EPISODE RESET: {RESET_NOTE} ==="

    def __init__(self, llm, history: int = 10, actions_per_call: int = 1, warn=print,
                 action_space: str = "primitive", intents_per_call: int = 1):
        if action_space not in ("primitive", "intent"):
            raise ValueError(f"action_space must be 'primitive' or 'intent', not {action_space!r}")
        self.llm = llm
        self.history = history
        self.action_space = action_space
        self.name = "react" if action_space == "intent" else "react-raw"
        self.per_call = intents_per_call if action_space == "intent" else actions_per_call
        self.actions_per_call = actions_per_call
        self.intents_per_call = intents_per_call
        self.warn = warn
        self.system = prompts.system_prompt(self.per_call, action_space)
        self.entries: list[str] = []   # formatted turns and reset markers
        self.episode = 1
        self.remind_format = False
        self._pending: ParsedResponse | None = None
        self._turn_index = 0
        self.repro_report: str | None = None    # set during an agent-driven reproduction attempt
        self.repro_steps_left = 0

    # -- history -------------------------------------------------------------
    def _window(self, n_turns: int) -> list[str]:
        """Last ``n_turns`` turns, plus any reset markers between them."""
        if n_turns <= 0:
            return []
        turn_idx = [i for i, e in enumerate(self.entries) if not self._is_marker(e)]
        if not turn_idx:
            return [e for e in self.entries if self._is_marker(e)][-1:]
        start = turn_idx[-n_turns] if len(turn_idx) >= n_turns else turn_idx[0]
        # keep a marker that sits immediately before the window start
        if start > 0 and self._is_marker(self.entries[start - 1]):
            start -= 1
        return self.entries[start:]

    @staticmethod
    def _is_marker(entry: str) -> bool:
        return entry.startswith("=== ")

    def on_episode_reset(self):
        self.episode += 1
        self.entries.append(self.RESET_MARKER)

    # -- agent-driven reproduction hooks (runner-level protocol, see repro.py) --
    def on_repro_start(self, report: str, steps: int):
        self.episode += 1
        self.entries.append(START_MARKER)
        self.repro_report, self.repro_steps_left = report, steps

    def on_repro_tick(self, steps_left: int):
        self.repro_steps_left = steps_left

    def on_repro_end(self):
        self.episode += 1
        self.entries.append(END_MARKER)
        self.repro_report = None

    def _turn_label(self, executed: list[int], results) -> str:
        """'[turn 12 | ep 3 | steps 14–18 (5 steps)]': episode steps, numbered like the
        observation's 'STEP: k/200' line (the step counter after each executed step)."""
        label = f"turn {self._turn_index} | ep {self.episode}"
        finals = [r.final for r in results or [] if r.final is not None]
        n = len(executed)
        if finals and n:
            last = finals[-1].state.get("episode_step")
            if last is not None:
                first = last - n + 1
                span = f"step {last}" if n == 1 else f"steps {first}–{last}"
                label += f" | {span} ({n} step{'s' if n != 1 else ''})"
        return f"[{label}]"

    def observe(self, executed: list[int], summary: str, results=None):
        p = self._pending or ParsedResponse()
        self._turn_index += 1
        head = self._turn_label(executed, results)
        if self.action_space == "intent":
            intents = "; ".join(r.intent for r in results or []) or "(none)"
            trace = " / ".join(r.trace for r in results or [])
            line = (f"{head} Thought: {p.thought or '(none)'} | "
                    f"Action: {intents} | Executor: {trace} | Result: {summary}")
        else:
            line = (f"{head} Thought: {p.thought or '(none)'} | "
                    f"Action: {','.join(map(str, executed))} | Result: {summary}")
        if p.bug_report:
            line += f" | Bug report: {p.bug_report}"
        self.entries.append(line)
        # entries older than the window are never shown again
        if len(self.entries) > 4 * max(self.history, 1) + 8:
            self.entries = self._window(self.history)

    # -- acting --------------------------------------------------------------
    def build_prompt(self, obs_text: str) -> tuple[str, int]:
        """Return (user_prompt, n_history_turns_used), applying the context guard."""
        n = self.history
        while True:
            block = (task_block(self.repro_report, self.repro_steps_left)
                     if self.repro_report is not None else None)
            user = prompts.turn_prompt("\n".join(self._window(n)), obs_text, self.episode,
                                       self.per_call, self.remind_format, self.action_space,
                                       task_block=block)
            limit = getattr(self.llm, "context_limit", None)
            if limit is None or estimate_tokens(self.system + user) <= limit:
                if n < self.history:
                    self.warn(f"[context guard] prompt too large for the context window; "
                              f"using {n}/{self.history} history turns")
                return user, n
            if n == 0:
                raise ContextOverflowError(
                    f"Even with no history the prompt (~{estimate_tokens(self.system + user)} "
                    f"tokens) exceeds 90% of the context window ({limit} tokens). "
                    "Increase --num-ctx.")
            n -= 1

    def act(self, obs_text: str, state: dict | None = None) -> Decision:
        user, n_used = self.build_prompt(obs_text)
        resp = self.llm.complete(self.system, user)
        if self.action_space == "intent":
            parsed = parse_intent_response(resp.text, self.intents_per_call)
            if parsed.parse_error:
                parsed.intents = [Intent("wait", "wait", action=WAIT, n=1)]
        else:
            parsed = parse_response(resp.text, self.actions_per_call)
            if parsed.parse_error:
                parsed.actions = [WAIT]
        self.remind_format = parsed.parse_error is not None
        self._pending = parsed
        record = {
            "system_prompt": self.system,
            "prompt": user,
            "response": resp.text,
            "parsed_thought": parsed.thought,
            "parsed_actions": parsed.actions,
            "parsed_bug_report": parsed.bug_report,
            "input_tokens": resp.input_tokens,
            "output_tokens": resp.output_tokens,
            "latency_s": round(resp.latency_s, 4),
            "parse_error": parsed.parse_error,
            "parse_note": parsed.parse_note,
            "history_turns_used": n_used,
            "parsed_reproduced": parsed.reproduced,
            "bug_report_discarded": parsed.bug_report_discarded,
        }
        if self.action_space == "intent":
            return Decision(llm_call=record, intents=list(parsed.intents))
        return Decision(list(parsed.actions), record)
