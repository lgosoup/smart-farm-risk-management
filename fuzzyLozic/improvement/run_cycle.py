from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from pathlib import Path as _Path
from config import load_config, DEFAULT_STATE_PATH

_DATA_DIR = _Path(__file__).resolve().parent.parent.parent / "data"

from .action_pool import build_default_action_pool
from .apply_action import apply_action_to_state
from .bandit import select_action_ucb1, update_reward
from .llm_gate import llm_gate
from .reward import compute_reward
from .state import ensure_action_stats, load_state, record_cycle, save_state, now_iso


def _read_jsonl_tail(path: str, n: int) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists() or n <= 0:
        return []
    rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows[-n:]


def run_improvement_cycle(
    *,
    state_path: str = DEFAULT_STATE_PATH,
    history_packets_path: str = str(_DATA_DIR / "history_packets.jsonl"),
    reward_window: int = 60,
    use_llm_gate: bool = True,
    vision_capture_dir: str = str(_DATA_DIR / "vision_captures"),
) -> Dict[str, Any]:
    """Run one improvement cycle.

    Flow (matches report):
      - (optional) Update bandit with the previous reward
      - Bandit selects ONE action from action pool
      - LLM gate approves or holds
      - If approved: apply to overrides (membership/rule weight)
      - Save state
    """
    cfg = load_config(state_path=state_path)  # includes existing overrides
    state = load_state(state_path)

    recent_packets = _read_jsonl_tail(history_packets_path, reward_window)
    current_packet = recent_packets[-1] if recent_packets else {}

    # 1) reward update for previous pending action (if any)
    pending = state.get("pending")
    reward_value = None
    reward_details: Dict[str, Any] = {}
    if isinstance(pending, dict) and isinstance(pending.get("action_id"), str):
        reward_value, reward_details = compute_reward(
            recent_packets=recent_packets,
            vision_capture_dir=vision_capture_dir,
        )
        update_reward(bandit_state=state["bandit"], action_id=str(pending["action_id"]), reward=float(reward_value))
        state["pending"] = None

    # 2) build action pool + ensure stats
    pool = build_default_action_pool(cfg)
    ensure_action_stats(state, pool)

    # 3) bandit select
    action = select_action_ucb1(actions=pool, bandit_state=state["bandit"])

    # 4) gate
    gate = llm_gate(current_packet=current_packet, recent_packets=recent_packets, action=action.to_dict()) if use_llm_gate else {
        "decision": "APPROVE",
        "reason": "LLM gate disabled",
        "bias_tags": [],
        "notes": [],
    }
    decision = str(gate.get("decision", "HOLD"))

    change: Dict[str, Any] = {"ok": False}
    if decision == "APPROVE":
        change = apply_action_to_state(cfg=cfg, state=state, action=action)
        if change.get("ok"):
            # mark as pending for next cycle reward update
            state["pending"] = {
                "action_id": action.action_id,
                "applied_at": now_iso(),
                "change": change,
            }
        else:
            decision = "HOLD"

    record_cycle(state, selected_action=action, decision=decision, gate=gate, reward=reward_value)
    save_state(state_path, state)

    return {
        "ts": now_iso(),
        "selected_action": action.to_dict(),
        "decision": decision,
        "gate": gate,
        "change": change,
        "reward_updated": reward_value is not None,
        "reward": reward_value,
        "reward_details": reward_details,
        "state_path": state_path,
    }


if __name__ == "__main__":
    # Run inside fuzzyLozic directory:
    #   python -m improvement.run_cycle
    result = run_improvement_cycle()
    print(json.dumps(result, ensure_ascii=False, indent=2))
