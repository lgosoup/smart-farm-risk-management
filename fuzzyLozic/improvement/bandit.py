from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Tuple

from .types import Action


def select_action_ucb1(
    *,
    actions: List[Action],
    bandit_state: Dict[str, Any],
) -> Action:
    """Conservative UCB1 selection.

    - First, explore untried actions.
    - Then, pick argmax( mean + c*sqrt(log(total)/n) ).
    """
    stats = bandit_state.get("actions", {}) if isinstance(bandit_state.get("actions"), dict) else {}
    c = float(bandit_state.get("c", 0.35))

    untried = [a for a in actions if int(stats.get(a.action_id, {}).get("n", 0)) <= 0]
    if untried:
        return random.choice(untried)

    total_n = sum(int(stats.get(a.action_id, {}).get("n", 0)) for a in actions)
    total_n = max(total_n, 1)

    best: Tuple[float, Action] | None = None
    for a in actions:
        s = stats.get(a.action_id, {}) if isinstance(stats.get(a.action_id, {}), dict) else {}
        n = int(s.get("n", 0))
        tr = float(s.get("total_reward", 0.0))
        mean = tr / n if n > 0 else 0.0
        bonus = c * math.sqrt(math.log(total_n + 1.0) / max(n, 1))
        score = mean + bonus
        if best is None or score > best[0]:
            best = (score, a)

    return best[1] if best else random.choice(actions)


def update_reward(
    *,
    bandit_state: Dict[str, Any],
    action_id: str,
    reward: float,
) -> None:
    stats = bandit_state.get("actions")
    if not isinstance(stats, dict):
        bandit_state["actions"] = {}
        stats = bandit_state["actions"]

    s = stats.get(action_id)
    if not isinstance(s, dict):
        s = {"n": 0, "total_reward": 0.0}
        stats[action_id] = s

    s["n"] = int(s.get("n", 0)) + 1
    s["total_reward"] = float(s.get("total_reward", 0.0)) + float(reward)
