
import json
import re
import hashlib
import sys
from pathlib import Path


def extract_state(observation):
    """Convert game observation text into a structured state."""

    position = re.search(
        r"YOU: at \((\d+),\s*(\d+)\), holding (.*)",
        observation
    )
    order = re.search(r"^ORDER:\s*(.*)$", observation, re.M)

    if position is None:
        raise ValueError("Could not find player position")

    def get_section(name, next_name):
        pattern = rf"^{name}\n(.*?)(?=^{next_name}|\Z)"
        match = re.search(pattern, observation, re.M | re.S)
        if not match:
            return []
        return [
            line.strip()
            for line in match.group(1).splitlines()
            if line.strip().startswith("- ")
        ]

    return {
        "x": int(position.group(1)),
        "y": int(position.group(2)),
        "holding": position.group(3).strip(),
        "order": order.group(1).strip() if order else None,
        "stations": get_section("STATIONS", "ITEMS"),
        "items": get_section("ITEMS", "ORDER:")
    }


def state_id(state):
    """Give identical states identical IDs."""
    content = json.dumps(state, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def build_graph(log_path):
    nodes = {}
    edges = []

    with open(log_path, encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)

            before = extract_state(row["state_before"])
            after = extract_state(row["state_after"])

            source = state_id(before)
            target = state_id(after)

            nodes[source] = before
            nodes[target] = after

            edges.append({
                "source": source,
                "target": target,
                "action": row["action"],
                "episode": row["episode"],
                "step": row["global_step"],
                "reward": row["reward"]
            })

    return {"nodes": nodes, "edges": edges}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python3 -m memory.observed_graph "
            "PATH_TO_STEPS_JSONL"
        )

    log_path = Path(sys.argv[1])
    graph = build_graph(log_path)

    output = log_path.parent / "observed_graph.json"
    output.write_text(json.dumps(graph, indent=2))

    print("Observed graph created!")
    print("Unique states:", len(graph["nodes"]))
    print("Transitions:", len(graph["edges"]))
    print("Saved to:", output)
