
import json
import sys
from pathlib import Path


def compare_graphs(run_dir):
    run_dir = Path(run_dir)

    observed = json.loads(
        (run_dir / "observed_graph.json").read_text()
    )
    reference = json.loads(
        (run_dir / "reference_graph.json").read_text()
    )

    observed_edges = {
        (e["episode"], e["step"]): e
        for e in observed["edges"]
    }

    matched = 0
    mismatched = 0
    unknown = 0

    for expected in reference["edges"]:
        key = (expected["episode"], expected["step"])
        actual = observed_edges.get(key)

        if actual is None:
            unknown += 1
        elif actual["target"] == expected["target"]:
            matched += 1
        else:
            mismatched += 1
            print(
                "Unexpected transition at step",
                expected["step"]
            )

    unknown += reference["unknown_transitions"]

    print("\nGraph comparison complete!")
    print("Expected and matched:", matched)
    print("Unexpected:", mismatched)
    print("Unknown:", unknown)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python3 -m memory.compare_graphs RUN_DIR"
        )

    compare_graphs(sys.argv[1])
