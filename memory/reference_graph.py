
import json
import re
import sys
from copy import deepcopy
from pathlib import Path

from .observed_graph import extract_state, state_id


# Overcooked Plus movement directions
DIRECTIONS = {
    0: (1, 0),    # Right
    1: (0, 1),    # Down
    2: (-1, 0),   # Left
    3: (0, -1)    # Up
}


def is_floor(observation, action):
    """Check if the action leads to an empty floor tile."""

    pattern = (
        rf"^\s*-\s*{action}\s+"
        r"(?:right|down|left|up)\s*"
        r"->\s*\(\d+,\s*\d+\):\s*floor\b"
    )

    return re.search(pattern, observation, re.M) is not None


def predict_next_state(observation, action):
    """Predict what should happen under normal movement rules."""

    if action not in DIRECTIONS:
        return None

    if not is_floor(observation, action):
        return None

    state = extract_state(observation)
    expected = deepcopy(state)

    dx, dy = DIRECTIONS[action]

    expected["x"] += dx
    expected["y"] += dy

# Move any item held by the player to the new position
    new_x = expected["x"]
    new_y = expected["y"]

    updated_items = []

    for item in expected["items"]:
        if "(held by you)" in item:
            item = re.sub(
                r"^\s*-\s*\(\d+,\s*\d+\)",
                f"- ({new_x},{new_y})",
                item
            )

        updated_items.append(item)

    expected["items"] = updated_items

    return expected


def build_reference_graph(log_path):
    nodes = {}
    edges = []
    unknown = 0

    with open(log_path, encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)

            before = extract_state(row["state_before"])

            expected = predict_next_state(
                row["state_before"],
                int(row["action"])
            )

            if expected is None:
                unknown += 1
                continue

            source = state_id(before)
            target = state_id(expected)

            nodes[source] = before
            nodes[target] = expected

            edges.append({
                "source": source,
                "target": target,
                "action": row["action"],
                "episode": row["episode"],
                "step": row["global_step"],
                "rule": "floor_movement"
            })

    return {
        "nodes": nodes,
        "edges": edges,
        "unknown_transitions": unknown
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python3 -m memory.reference_graph "
            "PATH_TO_STEPS_JSONL"
        )

    log_path = Path(sys.argv[1])
    graph = build_reference_graph(log_path)

    output = log_path.parent / "reference_graph.json"
    output.write_text(json.dumps(graph, indent=2))

    print("Reference graph created!")
    print("Unique states:", len(graph["nodes"]))
    print("Expected transitions:", len(graph["edges"]))
    print("Unknown transitions:", graph["unknown_transitions"])
    print("Saved to:", output)
