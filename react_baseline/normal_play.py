"""Order-independent normal-play bug sets: play every correct way to make the order (post hoc).

    python -m react_baseline.normal_play --task V2_task_07_tomato_steak
    python -m react_baseline.normal_play --all              # every task whose recipe is supported
    python -m react_baseline.normal_play --grid --write     # the 3 grid tasks -> configs/normal_play_bugs.yaml

The scripted solutions (configs/solutions/) follow one path per task, so the IDs they fire depend
on that path. The planner enumerates the correct-play variants instead, through the shared Action
reader on the real game:

  * preparation order: interleaved (prepare one ingredient, add it, prepare the next) or
    prep-first in every order (prepare everything, parking prepared vegetables on counters; a
    steak keeps cooking in the pan meanwhile);
  * assembly order: every order in which the ingredients go onto the plate;
  * plating method per ingredient: "scoop" (hold the plate and take the item from its station or
    counter into it), "carry" (pick the item up and put it down onto the plate on a counter), or
    "park_scoop" (pick it up, put it on a counter, then scoop it from there with the plate);
  * the plate held between consecutive scoops, or put down partway ("park" policy). In the
    interleaved schedule the plate is put down whenever your hands are needed.

Plates always start clean, and a correct delivery ends the episode, so no correct path washes a
plate (no washing variant). Oracle IDs are recorded on every step whose event did not fail (a
failed attempt is not normal play; there are none in a valid variant). A variant counts only if
the order is delivered and accepted. This is reporting only: it never changes ignore_bug_ids or
what the oracle counts.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from collections import Counter
from pathlib import Path

import yaml

from .action_reader import ActionReader, VisibleStep, bfs_path
from .solutions import NORMAL_PLAY_YAML, run_solution, solution_tasks

GRID_TASKS = ["V2_task_01_tomato", "V2_task_07_tomato_steak", "V2_task_11_tomato_lettuce_onion"]
CHOP = {"tomato", "lettuce", "onion", "fish"}   # chopped on the cutting board
PAN = {"steak"}                                 # chopped, then cooked in the pan
COOKER = {"rice"}                               # cooked in the rice cooker
SUPPORTED = CHOP | PAN | COOKER
STATION_METHODS = ("scoop", "carry", "park_scoop")
COUNTER_METHODS = ("scoop", "carry")


class PlanError(RuntimeError):
    pass


class Kitchen:
    """One episode on the real game, driven by intents through the shared Action reader."""

    def __init__(self, task: str):
        from .env_adapter import EnvAdapter
        self.adapter = EnvAdapter(task)
        self.adapter.reset()
        self.state = self.adapter.state
        self.reader = ActionReader()
        self.steps = 0
        self.ids: set[int] = set()          # oracle IDs on non-failed steps
        self.failed_events = 0
        self.delivered = False
        self.trace: list[str] = []

    def close(self):
        self.adapter.close()

    # -- low level --------------------------------------------------------------
    def do(self, intent: str):
        def step(a):
            r = self.adapter.step(a)
            self.steps += 1
            if (r.state.get("last_event") or {}).get("success") is False:
                self.failed_events += 1
            else:
                self.ids.update(int(b) for b in r.oracle["bugs"])
            self.delivered = self.delivered or r.oracle["delivery_success"]
            return VisibleStep(r.obs_text, r.summary, r.state, r.state["last_event"]["text"], r.done)

        res = self.reader.execute(intent, self.state, step)
        if res.final is not None:
            self.state = res.final.state
        self.trace.append(res.trace)
        if res.error or res.stop_reason not in ("completed", "episode_end"):
            raise PlanError(res.trace)
        if res.done and not self.delivered:
            raise PlanError(f"episode ended without a delivery: {res.trace}")
        return res

    def interact(self, pos):
        return self.do(f"interact({pos[0]},{pos[1]})")

    @property
    def held(self):
        return self.state["agent"]["holding"]

    def station(self, kind: str):
        hits = sorted(tuple(s["pos"]) for s in self.state["stations"] if s["kind"] == kind)
        if not hits:
            raise PlanError(f"no {kind} on this map")
        return hits[0]

    def station_holding(self, pos):
        for s in self.state["stations"]:
            if tuple(s["pos"]) == tuple(pos):
                return s["holding"]
        return None

    def counter_item(self, pos):
        for it in self.state["items"]:
            if it["where"] == "counter" and tuple(it["pos"]) == tuple(pos):
                return it
        return None

    def holding_at(self, pos):
        return self.station_holding(pos) if self.is_station(pos) else self.counter_item(pos)

    def is_station(self, pos):
        return any(tuple(s["pos"]) == tuple(pos) for s in self.state["stations"])

    def raw_source(self, name: str):
        hits = sorted(tuple(it["pos"]) for it in self.state["items"]
                      if it["where"] == "counter" and it["name"] == name
                      and not it.get("chopped") and not it.get("cooked"))
        if not hits:
            raise PlanError(f"no raw {name} on a counter")
        return hits[0]

    def plate_pos(self):
        hits = sorted(tuple(it["pos"]) for it in self.state["items"]
                      if it["name"] == "plate" and it["where"] == "counter")
        if not hits:
            raise PlanError("no plate on a counter")
        return hits[0]

    def free_counter(self):
        """Nearest empty counter next to a reachable floor tile (ties: top-left first)."""
        me, rows, best = tuple(self.state["agent"]["pos"]), self.state["map"], None
        for y, row in enumerate(rows):
            for x, ch in enumerate(row):
                if ch != "#":
                    continue
                adj = {(x + dx, y + dy) for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1))}
                path, _ = bfs_path(self.state, me, adj)
                if path is not None and (best is None or (len(path), y, x) < best[0]):
                    best = ((len(path), y, x), (x, y))
        if best is None:
            raise PlanError("no free counter")
        return best[1]

    # -- recipe operations ------------------------------------------------------------
    def fetch(self, name):
        self.interact(self.raw_source(name))
        if (self.held or {}).get("name") != name:
            raise PlanError(f"could not pick up {name}")

    def put_into(self, pos):
        self.interact(pos)
        if self.held is not None:
            raise PlanError(f"could not put {self.held['name']} at {pos}")

    def chop(self):
        board = self.station("knife")
        for _ in range(6):
            if (self.station_holding(board) or {}).get("chopped"):
                return board
            self.interact(board)
        raise PlanError("chopping did not finish")

    def wait_ready(self, pos):
        """Wait in place until the pan / rice cooker at ``pos`` holds cooked food."""
        kinds = {s["kind"] for s in self.state["stations"] if tuple(s["pos"]) == tuple(pos)}
        if not kinds & {"pan", "ricecooker", "oven"}:   # boards and counters hold finished items
            return
        for _ in range(60):
            if (self.station_holding(pos) or {}).get("cooked"):
                return
            self.do("wait")
        raise PlanError(f"food at {pos} never finished cooking")

    def take(self, pos):
        if self.held is not None:
            raise PlanError("hands not empty")
        self.interact(pos)
        if self.held is None:
            raise PlanError(f"could not take the item at {pos}")

    def park_held(self):
        pos = self.free_counter()
        self.put_into(pos)
        return pos

    def ensure_plate(self):
        if (self.held or {}).get("name") == "plate":
            return
        if self.held is not None:
            raise PlanError("holding food, not the plate")
        self.interact(self.plate_pos())
        if (self.held or {}).get("name") != "plate":
            raise PlanError("could not pick up the plate")

    def ensure_hands_free(self):
        if self.held is not None:
            if self.held["name"] != "plate":
                raise PlanError("holding food unexpectedly")
            self.park_held()

    def scoop(self, pos):
        self.ensure_plate()
        self.interact(pos)
        if self.holding_at(pos) is not None:
            raise PlanError(f"scooping from {pos} left the item there")

    def carry(self, pos):
        self.ensure_hands_free()
        self.take(pos)
        self.put_into(self.plate_pos())

    def deliver(self):
        self.ensure_plate()
        self.interact(self.station("delivery"))
        if not self.delivered:
            raise PlanError("delivery not accepted")

    # -- preparation --------------------------------------------------------------
    def prepare(self, name):
        """Prepare one ingredient; -> where it now is (station position)."""
        self.ensure_hands_free()
        if name in COOKER:
            self.fetch(name)
            cooker = self.station("ricecooker")
            self.put_into(cooker)
            return cooker
        self.fetch(name)
        self.put_into(self.station("knife"))
        board = self.chop()
        if name in PAN:
            self.take(board)
            pan = self.station("pan")
            self.put_into(pan)
            return pan
        return board

    def add(self, name, where, method):
        """Put the prepared ingredient at ``where`` onto the plate with ``method``."""
        if method == "scoop":
            if self.held is not None and self.held["name"] != "plate":
                raise PlanError("holding food, not the plate")
            self.wait_ready(where)
            self.scoop(where)
        elif method == "carry":
            self.ensure_hands_free()
            self.wait_ready(where)
            self.carry(where)
        elif method == "park_scoop":
            self.ensure_hands_free()
            self.wait_ready(where)
            self.take(where)
            spot = self.park_held()
            self.scoop(spot)
        else:
            raise ValueError(method)


# ---------------------------------------------------------------------------- variants
def recipe(task: str) -> list[str]:
    from .env_adapter import EnvAdapter
    a = EnvAdapter(task)
    try:
        a.reset()
        orders = a.state["order"]
    finally:
        a.close()
    if len(orders) != 1:
        raise PlanError(f"{task}: {len(orders)} orders; the planner handles one")
    return list(orders[0])


def variants(ingredients: list[str]) -> list[dict]:
    """Every variant: schedule, preparation order, assembly order, method per ingredient,
    plate policy (only where it can matter: two or more scoops in a row while prepping first)."""
    out = []
    n = len(ingredients)
    for assembly in itertools.permutations(ingredients):
        for methods in itertools.product(STATION_METHODS, repeat=n):
            out.append({"schedule": "interleaved", "prep": list(assembly), "assembly": list(assembly),
                        "methods": list(methods), "plate": "as_needed"})
    for prep in itertools.permutations(ingredients):
        for assembly in itertools.permutations(ingredients):
            choices = [STATION_METHODS if (i in PAN or i in COOKER) else COUNTER_METHODS for i in assembly]
            for methods in itertools.product(*choices):
                scoops = sum(1 for a, b in zip(methods, methods[1:]) if a == "scoop" and b == "scoop")
                for plate in (("keep", "park") if scoops else ("keep",)):
                    out.append({"schedule": "prep_first", "prep": list(prep), "assembly": list(assembly),
                                "methods": list(methods), "plate": plate})
    return out


def variant_name(v: dict) -> str:
    plating = ",".join(f"{i}:{m}" for i, m in zip(v["assembly"], v["methods"]))
    name = f"{v['schedule']}; plate {plating}"
    if v["schedule"] == "prep_first":
        name += f"; prep {'>'.join(v['prep'])}; plate held: {v['plate']}"
    return name


def play_variant(task: str, v: dict) -> dict:
    k = Kitchen(task)
    error = None
    try:
        where = {}
        if v["schedule"] == "interleaved":
            for i, (name, method) in enumerate(zip(v["assembly"], v["methods"])):
                where[name] = k.prepare(name)
                k.add(name, where[name], method)
        else:
            for name in v["prep"]:
                spot = k.prepare(name)
                if name in CHOP:            # free the board: park the chopped item on a counter
                    k.take(spot)
                    spot = k.park_held()
                where[name] = spot
            for i, (name, method) in enumerate(zip(v["assembly"], v["methods"])):
                k.add(name, where[name], method)
                nxt = v["methods"][i + 1] if i + 1 < len(v["methods"]) else None
                if v["plate"] == "park" and method == "scoop" and nxt == "scoop":
                    k.park_held()
        k.deliver()
    except PlanError as exc:
        error = str(exc)
    finally:
        k.close()
    return {"name": variant_name(v), **v, "delivered": k.delivered and error is None, "error": error,
            "steps": k.steps, "failed_events": k.failed_events, "bug_ids_raw": sorted(k.ids)}


def plan_task(task: str, progress=None) -> dict:
    ings = recipe(task)
    unsupported = sorted(set(ings) - SUPPORTED)
    if unsupported:
        return {"task": task, "ingredients": ings, "unsupported": unsupported}
    rows = []
    for v in variants(ings):
        rows.append(play_variant(task, v))
        if progress:
            progress(rows[-1])
    ok = [r for r in rows if r["delivered"]]
    counts = Counter(b for r in ok for b in r["bug_ids_raw"])
    return {
        "task": task, "ingredients": ings, "variants": len(rows), "variants_delivered": len(ok),
        "union": sorted(counts),
        "always": sorted(b for b, c in counts.items() if c == len(ok)),
        "id_variant_counts": {int(b): c for b, c in sorted(counts.items())},
        "rows": rows,
    }


# ---------------------------------------------------------------------------- YAML
HEADER = """\
# Oracle bug IDs (unfiltered) that fire during correct play of each task. Post-hoc reporting only:
# used for summary.unique_bugs_excluding_normal_play and the *_adj metrics (react_baseline.rescore);
# never changes ignore_bug_ids or what the oracle counts.
#   bug_ids_raw        the set the metrics use. source: planner = union over every correct-play
#                      variant (python -m react_baseline.normal_play --grid --write); source:
#                      scripted_solution = the one scripted path (python -m react_baseline.solutions --write).
#   always             IDs that fire in every delivered variant
#   id_variant_counts  in how many delivered variants each ID fires
#   per_variant        variant -> IDs (failed variants list their error instead)
#   scripted_solution  the IDs of configs/solutions/<task>.yaml (kept for comparison)
"""


def load_yaml(path=None) -> dict:
    path = Path(path or NORMAL_PLAY_YAML)
    if not path.exists():
        return {"tasks": {}}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {"tasks": {}}


class _Flow(list):
    pass


yaml.SafeDumper.add_representer(_Flow, lambda d, v: d.represent_sequence("tag:yaml.org,2002:seq", v,
                                                                         flow_style=True))


def write_yaml(data: dict, path=None):
    path = Path(path or NORMAL_PLAY_YAML)
    tasks = {}
    for t, e in sorted((data.get("tasks") or {}).items()):
        e = dict(e)
        for key in ("bug_ids_raw", "always", "ingredients"):
            if key in e:
                e[key] = _Flow(e[key])
        if "per_variant" in e:
            e["per_variant"] = {k: (_Flow(v) if isinstance(v, list) else v) for k, v in e["per_variant"].items()}
        if "scripted_solution" in e:
            e["scripted_solution"] = {**e["scripted_solution"],
                                      "bug_ids_raw": _Flow(e["scripted_solution"]["bug_ids_raw"])}
        tasks[t] = e
    path.write_text(HEADER + yaml.safe_dump({"tasks": tasks}, sort_keys=False, width=110), encoding="utf-8")


def planner_entry(result: dict, scripted: dict | None) -> dict:
    entry = {
        "source": "planner",
        "bug_ids_raw": result["union"],
        "ingredients": result["ingredients"],
        "variants": result["variants"],
        "variants_delivered": result["variants_delivered"],
        "always": result["always"],
        "id_variant_counts": result["id_variant_counts"],
    }
    if scripted is not None:
        entry["scripted_solution"] = {"bug_ids_raw": scripted["bug_ids_raw"], "steps": scripted["steps"]}
    entry["per_variant"] = {r["name"]: (r["bug_ids_raw"] if r["delivered"] else f"FAILED: {r['error']}")
                            for r in result["rows"]}
    return entry


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--task", action="append", help="task id (repeatable)")
    g.add_argument("--all", action="store_true", help="every task whose recipe the planner supports")
    g.add_argument("--grid", action="store_true", help="the 3 grid tasks")
    p.add_argument("--write", action="store_true", help=f"update {NORMAL_PLAY_YAML.name} for these tasks")
    p.add_argument("--verbose", action="store_true", help="print every variant")
    args = p.parse_args(argv)
    if args.all:
        from .env_adapter import list_tasks
        tasks = list_tasks()
    else:
        tasks = GRID_TASKS if args.grid else args.task
    data, ok = load_yaml(), True
    data.setdefault("tasks", {})
    for task in tasks:
        res = plan_task(task, progress=(lambda r: print(f"  {'ok ' if r['delivered'] else 'BAD'} "
                                                        f"{r['steps']:>3} steps {r['bug_ids_raw']} {r['name']}"
                                                        + (f" -- {r['error']}" if r['error'] else "")))
                        if args.verbose else None)
        if "unsupported" in res:
            print(f"{task}: skipped (recipe {res['ingredients']}; planner does not handle {res['unsupported']})")
            continue
        print(f"{task}: {res['variants_delivered']}/{res['variants']} variants delivered; "
              f"union {res['union']}; in every variant {res['always']}")
        bad = [r for r in res["rows"] if not r["delivered"]]
        for r in bad[:5]:
            print(f"  not delivered: {r['name']}: {r['error']}")
        ok &= res["variants_delivered"] > 0
        if args.write:
            scripted = run_solution(task) if task in solution_tasks() else None
            data["tasks"][task] = planner_entry(res, scripted)
    if args.write:
        write_yaml(data)
        print(f"wrote {NORMAL_PLAY_YAML}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
