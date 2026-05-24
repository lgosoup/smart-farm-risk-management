from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .types import Action


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_state(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {
            "version": 1,
            "created_at": now_iso(),
            "overrides": {"memberships": {}, "rule_weights": {}},
            "bandit": {"algo": "ucb1", "c": 0.35, "actions": {}},
            "pending": None,
            "last_cycle": None,
        }
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            # minimal schema hardening
            obj.setdefault("version", 1)
            obj.setdefault("overrides", {"memberships": {}, "rule_weights": {}})
            obj.setdefault("bandit", {"algo": "ucb1", "c": 0.35, "actions": {}})
            obj.setdefault("pending", None)
            obj.setdefault("last_cycle", None)
            return obj
    except Exception:
        pass
    return {
        "version": 1,
        "created_at": now_iso(),
        "overrides": {"memberships": {}, "rule_weights": {}},
        "bandit": {"algo": "ucb1", "c": 0.35, "actions": {}},
        "pending": None,
        "last_cycle": None,
    }


def save_state(path: str, state: Dict[str, Any]) -> None:
    p = Path(path)
    p.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_action_stats(state: Dict[str, Any], actions: List[Action]) -> None:
    bandit = state.get("bandit")
    if not isinstance(bandit, dict):
        state["bandit"] = {"algo": "ucb1", "c": 0.35, "actions": {}}
        bandit = state["bandit"]

    stats = bandit.get("actions")
    if not isinstance(stats, dict):
        bandit["actions"] = {}
        stats = bandit["actions"]

    for a in actions:
        if a.action_id not in stats:
            stats[a.action_id] = {"n": 0, "total_reward": 0.0}


def record_cycle(
    state: Dict[str, Any],
    *,
    selected_action: Action,
    decision: str,
    gate: Dict[str, Any],
    reward: Optional[float],
) -> None:
    state["last_cycle"] = {
        "ts": now_iso(),
        "selected_action": selected_action.to_dict(),
        "decision": decision,
        "gate": gate,
        "reward": reward,
    }
