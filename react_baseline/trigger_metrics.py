"""Oracle-side trigger evaluation (analysis only; nothing here ever reaches an agent).

A fire is "useful" when its transition triggered at least one adjusted bug (oracle IDs minus the
task's normal-play set, after ignore_bug_ids). Given the run's transitions (global step range,
whether the memory check ran and fired) and steps.jsonl rows:

  precision_adj       share of fires whose transition triggered >= 1 adjusted bug
  recall_unique_adj   share of unique adjusted bugs whose first-occurrence transition had a fire
  recall_events_adj   share of all adjusted bug occurrences inside fired transitions

First occurrences before the memory check runs (warm-up) count as missed. With ``classes`` (bug ID ->
proposed RQ3 class from the draft taxonomy) recall_unique_adj is also split into
recall_unique_adj_A (A_transition_anomaly) and recall_unique_adj_B (B_no_transition_anomaly).
"""
from __future__ import annotations

CLASS_A, CLASS_B = "A_transition_anomaly", "B_no_transition_anomaly"


def evaluate(transitions: list[dict], step_rows: list[dict], normal, classes: dict | None = None) -> dict:
    """``transitions``: dicts with start, end (global steps, inclusive), checked, fired.
    ``normal``: the task's normal-play IDs (None: unknown task -> all fields None)."""
    keys = ("precision_adj", "recall_unique_adj", "recall_events_adj")
    if normal is None:
        return {k: None for k in keys} | {"fires": None, "checks": None, "adjusted_bugs": None}
    normal = set(normal)
    owner = {}
    for i, t in enumerate(transitions):
        for g in range(t["start"], t["end"] + 1):
            owner[g] = i
    occ = [(r["global_step"], b) for r in step_rows for b in r["bug_ids_triggered"] if b not in normal]
    by_t: dict[int, set] = {}
    first: dict[int, int] = {}
    for g, b in occ:
        if g in owner:
            by_t.setdefault(owner[g], set()).add(b)
        first.setdefault(b, g)
    fired = {i for i, t in enumerate(transitions) if t["fired"]}
    checks = sum(1 for t in transitions if t["checked"])
    hit_unique = sum(1 for b, g in first.items() if owner.get(g) in fired)
    hit_events = sum(1 for g, b in occ if owner.get(g) in fired)
    split = {}
    if classes is not None:
        for tag, cls in (("A", CLASS_A), ("B", CLASS_B)):
            ids = [b for b in first if classes.get(b) == cls]
            hits = sum(1 for b in ids if owner.get(first[b]) in fired)
            split[f"recall_unique_adj_{tag}"] = (hits / len(ids)) if ids else None
            split[f"adjusted_bugs_{tag}"] = len(ids)
    return {
        "precision_adj": (sum(1 for i in fired if by_t.get(i)) / len(fired)) if fired else None,
        "recall_unique_adj": (hit_unique / len(first)) if first else None,
        "recall_events_adj": (hit_events / len(occ)) if occ else None,
        "fires": len(fired), "checks": checks, "adjusted_bugs": len(first),
        **split,
    }
