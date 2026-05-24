from __future__ import annotations

from typing import Dict, List

from .types import Action


def build_default_action_pool(cfg: Dict[str, object]) -> List[Action]:
    """Curated action pool.

    Principles:
    - Actions are small, single-step adjustments.
    - No abstract templates; values are fixed.
    - Only touch two levers: membership points, and per-rule weights.
    """

    actions: List[Action] = []

    # -------- Membership micro-shifts --------
    # Water temperature (°C)
    for delta in (-0.2, +0.2):
        actions += [
            Action(
                action_id=f"M_WT_OPT_P1_{'DN' if delta < 0 else 'UP'}",
                action_type="membership_points",
                var_key="water_temperature",
                shape_label="안정(최적)",
                point_index=1,
                delta=delta,
                description=f"water_temperature 안정(최적) trap P1 {delta:+}°C",
            ),
            Action(
                action_id=f"M_WT_OPT_P2_{'DN' if delta < 0 else 'UP'}",
                action_type="membership_points",
                var_key="water_temperature",
                shape_label="안정(최적)",
                point_index=2,
                delta=delta,
                description=f"water_temperature 안정(최적) trap P2 {delta:+}°C",
            ),
            Action(
                action_id=f"M_WT_BORDER_P0_{'DN' if delta < 0 else 'UP'}",
                action_type="membership_points",
                var_key="water_temperature",
                shape_label="경계(상승)",
                point_index=0,
                delta=delta,
                description=f"water_temperature 경계(상승) tri P0 {delta:+}°C",
            ),
        ]

    # Flow ratio (unitless)
    for delta in (-0.03, +0.03):
        actions += [
            Action(
                action_id=f"M_FR_STABLE_P1_{'DN' if delta < 0 else 'UP'}",
                action_type="membership_points",
                var_key="flow_ratio",
                shape_label="안정",
                point_index=1,
                delta=delta,
                description=f"flow_ratio 안정 trap P1 {delta:+}",
            ),
            Action(
                action_id=f"M_FR_STABLE_P2_{'DN' if delta < 0 else 'UP'}",
                action_type="membership_points",
                var_key="flow_ratio",
                shape_label="안정",
                point_index=2,
                delta=delta,
                description=f"flow_ratio 안정 trap P2 {delta:+}",
            ),
            Action(
                action_id=f"M_FR_LOW_P1_{'DN' if delta < 0 else 'UP'}",
                action_type="membership_points",
                var_key="flow_ratio",
                shape_label="낮음(주의)",
                point_index=1,
                delta=delta,
                description=f"flow_ratio 낮음(주의) tri P1 {delta:+}",
            ),
        ]

    # EC (water mode)
    for delta in (-0.03, +0.03):
        actions += [
            Action(
                action_id=f"M_EC_W_STABLE_P1_{'DN' if delta < 0 else 'UP'}",
                action_type="membership_points",
                var_key="EC_water",
                shape_label="안정",
                point_index=1,
                delta=delta,
                description=f"EC(water) 안정 trap P1 {delta:+}",
            ),
            Action(
                action_id=f"M_EC_W_HIGH_P1_{'DN' if delta < 0 else 'UP'}",
                action_type="membership_points",
                var_key="EC_water",
                shape_label="높음(경고)",
                point_index=1,
                delta=delta,
                description=f"EC(water) 높음(경고) tri P1 {delta:+}",
            ),
        ]

    # Turbidity (NTU)
    for delta in (-0.5, +0.5):
        actions += [
            Action(
                action_id=f"M_TURB_CLEAR_P2_{'DN' if delta < 0 else 'UP'}",
                action_type="membership_points",
                var_key="turbidity",
                shape_label="맑음(안정)",
                point_index=2,
                delta=delta,
                description=f"turbidity 맑음(안정) trap P2 {delta:+}NTU",
            )
        ]

    # -------- Rule weight micro-mults --------
    for mult in (0.99, 1.01):
        for rid in ["R6", "R7", "R8", "R9", "R10", "R11", "R16", "R17", "R18", "R19"]:
            actions.append(
                Action(
                    action_id=f"W_{rid}_{'DN' if mult < 1.0 else 'UP'}",
                    action_type="rule_weight",
                    rule_id=rid,
                    multiplier=mult,
                    description=f"Rule {rid} weight ×{mult}",
                )
            )

    seen = set()
    unique: List[Action] = []
    for a in actions:
        if a.action_id in seen:
            continue
        seen.add(a.action_id)
        unique.append(a)
    return unique
