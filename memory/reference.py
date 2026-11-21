"""Reference expectations from the game manual (configs/reference_rules.yaml).

The rules are written only from the GAME RULES text of the system prompt plus Tom's floor-movement
rule (see the YAML header); bug_manager.py, the taxonomy and normal_play_bugs were not consulted.
They are keyed on the TKG's ctx0 abstraction (intent kind | hands:<held> | target entity) and predict
the outcome (event type, success, held after, target after). ``Reference.lookup(ctx0)`` returns the
first matching rule (or None: unknown situation); ``hit.matches(outcome)`` checks an observed outcome.

    python -m memory.reference --coverage runs/trigger_eval/20261027-005617/runs

reports, over stored cheap trajectories, the share of transitions a rule covers, the share of
covered ones whose outcome matched, and the most frequent mismatches (rule errors or places where
the game differs from the manual). This ports Tom's reference_graph / compare_graphs
(origin/cheap-agent-baseline) to the live TKG: rules on abstract keys instead of predicted text states.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml

from .abstraction import abstract

RULES_YAML = Path(__file__).resolve().parent.parent / "configs" / "reference_rules.yaml"
_SEP = " | "
_SUCCESS = {"ok": r"True|-", "any": r".*", "True": "True", "False": "False"}


@dataclass
class Rule:
    id: str
    intents: tuple
    held: re.Pattern
    target: re.Pattern
    expect: dict
    changes_state: bool
    manual: str
    says: str


@dataclass
class RuleHit:
    rule: Rule
    vars: dict

    def _pattern(self, template: str) -> re.Pattern:
        def fill(m):
            return re.escape(self.vars.get(m.group(1), m.group(0)))
        return re.compile(re.sub(r"\{(\w+)\}", fill, template))

    def matches(self, outcome: str) -> bool:
        event, success, hands, target = outcome.split(_SEP)
        e = self.rule.expect
        held = hands[len("hands:"):] if hands.startswith("hands:") else hands
        return (event == str(e["event"]) and re.fullmatch(_SUCCESS[str(e.get("success", "any"))], success) is not None
                and self._pattern(e["held_after"]).fullmatch(held) is not None
                and self._pattern(e["target_after"]).fullmatch(target) is not None)


class Reference:
    def __init__(self, path: Path | str = RULES_YAML):
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        self.rules = [Rule(r["id"], tuple(r["when"]["intent"]), re.compile(r["when"]["held"]),
                           re.compile(r["when"]["target"]), r["expect"], bool(r["changes_state"]),
                           r.get("manual", ""), r.get("says", ""))
                      for r in data["rules"]]
        self.manual_mismatches = data.get("manual_mismatches") or []
        self.by_id = {r.id: r for r in self.rules}

    def lookup(self, ctx0: str) -> RuleHit | None:
        kind, hands, target = ctx0.split(_SEP)
        held = hands[len("hands:"):] if hands.startswith("hands:") else hands
        for r in self.rules:
            if kind not in r.intents:
                continue
            mh, mt = r.held.fullmatch(held), r.target.fullmatch(target)
            if mh and mt:
                v = {**mh.groupdict(), **mt.groupdict(), "held": held, "target": target}
                if v.get("item"):
                    v["item_chopped"] = v["item"].replace("unchopped", "chopped", 1)
                return RuleHit(r, {k: x for k, x in v.items() if x is not None})
        return None

    def verdict(self, ctx0: str, outcome: str) -> tuple[str | None, str]:
        """-> (rule_id | None, "match" | "mismatch" | "none")."""
        hit = self.lookup(ctx0)
        if hit is None:
            return None, "none"
        return hit.rule.id, ("match" if hit.matches(outcome) else "mismatch")


_DEFAULT: Reference | None = None


def default_reference() -> Reference:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Reference()
    return _DEFAULT


# ---------------------------------------------------------------------------- coverage report
def stored_transitions(run_dir: Path) -> list[dict]:
    """Abstract transitions of a stored run (transitions.jsonl), in order."""
    out, prev = [], {}
    for line in (run_dir / "transitions.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        ev = r["event_records"][-1] if r["event_records"] else None
        k = abstract(r["kind"], ev, prev.get(r["episode"]))
        prev[r["episode"]] = k["outcome"]
        out.append(k)
    return out


def coverage(runs_root: Path, ref: Reference | None = None) -> dict:
    ref = ref or Reference()
    n = covered = matched = 0
    per_rule, mism = Counter(), Counter()
    uncovered = Counter()
    for d in sorted(p for p in runs_root.iterdir() if (p / "transitions.jsonl").exists()):
        for t in stored_transitions(d):
            n += 1
            rid, verdict = ref.verdict(t["ctx0"], t["outcome"])
            if rid is None:
                uncovered[t["ctx0"]] += 1
                continue
            covered += 1
            per_rule[(rid, verdict)] += 1
            if verdict == "match":
                matched += 1
            else:
                mism[(rid, t["ctx0"], t["outcome"])] += 1
    return {"transitions": n, "covered": covered, "matched": matched,
            "coverage": covered / n if n else None, "match_rate": matched / covered if covered else None,
            "per_rule": {f"{r}:{v}": c for (r, v), c in sorted(per_rule.items())},
            "top_mismatches": [{"rule": r, "ctx0": c, "observed": o, "count": k}
                               for (r, c, o), k in mism.most_common(25)],
            "top_uncovered": [{"ctx0": c, "count": k} for c, k in uncovered.most_common(15)]}


def main(argv=None):
    p = argparse.ArgumentParser(description="Reference-rule coverage on stored trajectories")
    p.add_argument("--coverage", required=True, help="folder of run dirs with transitions.jsonl")
    p.add_argument("--json", default=None, help="also write the report here")
    a = p.parse_args(argv)
    rep = coverage(Path(a.coverage))
    print(f"transitions {rep['transitions']}, covered {rep['covered']} ({rep['coverage']:.1%}), "
          f"matched {rep['matched']} ({rep['match_rate']:.1%} of covered)")
    print("per rule:", json.dumps(rep["per_rule"], indent=1))
    print("top mismatches:")
    for m in rep["top_mismatches"]:
        print(f"  {m['count']:>4}  [{m['rule']}]  {m['ctx0']}  ->  {m['observed']}")
    print("top uncovered:")
    for m in rep["top_uncovered"]:
        print(f"  {m['count']:>4}  {m['ctx0']}")
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
