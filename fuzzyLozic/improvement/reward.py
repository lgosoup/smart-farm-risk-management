from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent

log = logging.getLogger(__name__)


def _import_vision():
    root = str(_ROOT_DIR)
    if root not in sys.path:
        sys.path.insert(0, root)
    from smartfarm.vision.plant_health_analyzer import compute_vision_reward
    return compute_vision_reward


def compute_reward(
    *,
    recent_packets: List[Dict[str, Any]],
    vision_capture_dir: str = str(_DATA_DIR / "vision_captures"),
    target_alarm_rate: Tuple[float, float] = (0.05, 0.25),
) -> Tuple[float, Dict[str, Any]]:
    """Compute reward for the previous cycle.

    Priority 1: webcam plant health analysis (vision score) if today's image exists.
    Priority 2: heuristic proxy (alarm rate + stability) when image unavailable.

    Returns (reward, details).
    """
    # Priority 1: vision-based reward
    try:
        compute_vision_reward = _import_vision()
        vision_result = compute_vision_reward(vision_capture_dir)
        if vision_result is not None:
            reward, details = vision_result
            log.info("비전 보상 계산 완료: composite=%.4f reward=%.4f", details.get("composite", 0), reward)
            return reward, details
    except Exception as e:
        log.warning("비전 보상 계산 실패 (%s) — 휴리스틱으로 fallback", e)

    # Priority 2: heuristic fallback
    labels: List[str] = []
    scores: List[float] = []
    for p in recent_packets:
        ak = p.get("action_key")
        if isinstance(ak, str):
            labels.append(ak)
        risk = p.get("risk") if isinstance(p.get("risk"), dict) else {}
        rs = risk.get("risk_score")
        if isinstance(rs, (int, float)):
            scores.append(float(rs))

    n = max(len(labels), 1)
    alarm_like = sum(1 for x in labels if x in {"경고", "위험", "긴급"})
    alarm_rate = alarm_like / n

    # stability: penalize large score jumps
    diffs = []
    for a, b in zip(scores, scores[1:]):
        diffs.append(abs(b - a))
    mean_jump = sum(diffs) / max(len(diffs), 1)

    lo, hi = target_alarm_rate
    rate_penalty = 0.0
    if alarm_rate < lo:
        rate_penalty = (lo - alarm_rate) * 4.0
    elif alarm_rate > hi:
        rate_penalty = (alarm_rate - hi) * 4.0

    jump_penalty = (mean_jump / 20.0)  # normalize
    reward = 1.0 - rate_penalty - jump_penalty

    return reward, {
        "source": "heuristic",
        "alarm_rate": alarm_rate,
        "mean_risk_jump": mean_jump,
        "rate_penalty": rate_penalty,
        "jump_penalty": jump_penalty,
    }
