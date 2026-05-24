from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional


ActionType = Literal["membership_points", "rule_weight"]


@dataclass(frozen=True)
class Action:
    """A single, fully-specified micro-adjustment (no templates).

    This matches the report definition: "액션 하나 = 수치까지 확정된 단일 변경".
    """

    action_id: str
    action_type: ActionType

    # membership_points
    var_key: Optional[str] = None  # e.g., "T", "DO", "EC_water"
    shape_label: Optional[str] = None  # e.g., "안정(최적)"
    point_index: Optional[int] = None  # 0..2 (tri) or 0..3 (trap)
    delta: Optional[float] = None

    # rule_weight
    rule_id: Optional[str] = None
    multiplier: Optional[float] = None

    # Human-readable description
    description: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "action_id": self.action_id,
            "action_type": self.action_type,
            "var_key": self.var_key,
            "shape_label": self.shape_label,
            "point_index": self.point_index,
            "delta": self.delta,
            "rule_id": self.rule_id,
            "multiplier": self.multiplier,
            "description": self.description,
        }
