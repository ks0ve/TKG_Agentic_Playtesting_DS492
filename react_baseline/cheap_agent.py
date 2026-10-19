
import random
import re
from collections import Counter

from .agents import Decision


class CheapAgent:
    name = "cheap"

    def __init__(self, seed=0):
        # Use a private random generator for reproducibility.
        self.rng = random.Random(seed)

        # Remember visited positions.
        self.visits = Counter()

        # Remember attempted actions at each position.
        self.attempts = Counter()

    def act(self, obs_text):
        # STEP A: Find the player's position.
        match = re.search(
            r"YOU: at \((\d+),\s*(\d+)\)",
            obs_text
        )

        if match is None:
            return Decision([self.rng.randrange(4)])

        x, y = map(int, match.groups())
        self.visits[(x, y)] += 1

        # STEP B: Read neighboring tiles.
        choices = []

        for line in obs_text.splitlines():
            neighbor = re.match(
                r"\s*-\s*([0-3])\s+"
                r"(right|down|left|up)\s*"
                r"->\s*\((\d+),\s*(\d+)\):\s*(.*)",
                line
            )

            if neighbor is None:
                continue

            action = int(neighbor.group(1))
            nx = int(neighbor.group(3))
            ny = int(neighbor.group(4))
            description = neighbor.group(5).lower()

            key = (x, y, action)

            # STEP C: Score possible actions.
            if "floor" in description:
                # Prefer less-visited floor tiles.
                score = 3 / (1 + self.visits[(nx, ny)])
            else:
                # Explore different interactions.
                score = 1 / (1 + self.attempts[key])

            # Add a small penalty for repeating actions.
            score -= 0.1 * self.attempts[key]

            choices.append((score, action))

        # STEP D: Choose the highest-scoring action.
        if not choices:
            return Decision([self.rng.randrange(4)])

        best_score = max(score for score, _ in choices)

        best_actions = [
            action
            for score, action in choices
            if abs(score - best_score) < 1e-9
        ]

        chosen = self.rng.choice(best_actions)
        self.attempts[(x, y, chosen)] += 1

        return Decision([chosen])

    def observe(self, executed, summary):
        # Later: use the result to improve decisions.
        pass

    def on_episode_reset(self):
        # Keep exploration counters across episodes.
        pass
