"""Draft bug taxonomy: measured alongside the oracle, never used to filter.

``configs/bug_taxonomy_draft.csv`` has one row per SAGE bug ID (family,
reachability, trigger signature, proposed RQ3 class), all ``status=proposed``.
This module only counts a run's unique bug IDs by class and family for
summary.json. It does not change ``ignore_bug_ids``, metrics or behaviour, and
nothing here ever reaches the agent.
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

TAXONOMY_CSV = Path(__file__).resolve().parent.parent / "configs" / "bug_taxonomy_draft.csv"
UNREACHABLE = "unreachable_on_shipped_maps"


def load_taxonomy(path: Path | str = TAXONOMY_CSV) -> dict[int, dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return {int(r["bug_id"]): r for r in csv.DictReader(f)}


def taxonomy_counts(bug_ids, taxonomy: dict[int, dict] | None = None) -> dict:
    """-> {bugs_by_proposed_class, bugs_by_family, taxonomy_warnings} for ``bug_ids``."""
    if taxonomy is None:
        try:
            taxonomy = load_taxonomy()
        except OSError as exc:
            return {"bugs_by_proposed_class": None, "bugs_by_family": None,
                    "taxonomy_warnings": [f"taxonomy not loaded: {exc}"]}
    by_class, by_family, warnings = Counter(), Counter(), []
    for b in sorted(set(int(x) for x in bug_ids)):
        row = taxonomy.get(b)
        if row is None:
            by_class["unknown"] += 1
            by_family["unknown"] += 1
            warnings.append(f"bug {b} is not in the draft taxonomy")
            continue
        by_class[row["proposed_rq3_class"] or "unknown"] += 1
        by_family[row["family"] or "unknown"] += 1
        if row["reachability"] == UNREACHABLE:
            warnings.append(f"bug {b} ({row['name']}) fired, but the draft taxonomy marks it "
                            f"{UNREACHABLE}")
    return {"bugs_by_proposed_class": dict(sorted(by_class.items())),
            "bugs_by_family": dict(sorted(by_family.items())),
            "taxonomy_warnings": warnings}
