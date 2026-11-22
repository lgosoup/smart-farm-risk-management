from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .types import Action


def _get_var_shapes(cfg: Dict[str, object], var_key: str):
    variables = cfg.get("variables")
    if not isinstance(variables, dict):
        return None

    if var_key == "EC_water":
        ec = variables.get("EC")
        if isinstance(ec, dict):
            return getattr(ec.get("water"), "shapes", None)
        return None
    if var_key == "EC_hydro":
        ec = variables.get("EC")
        if isinstance(ec, dict):
            return getattr(ec.get("hydro"), "shapes", None)
        return None

    obj = variables.get(var_key)
    return getattr(obj, "shapes", None)


def _clamp_monotonic(points: List[float], idx: int, eps: float = 1e-6) -> List[float]:
    """Ensure points are non-decreasing with minimal clamping."""
    pts = list(points)
    # clamp idx within neighbors
    if idx > 0:
        pts[idx] = max(pts[idx], pts[idx - 1] + eps)
    if idx < len(pts) - 1:
        pts[idx] = min(pts[idx], pts[idx + 1] - eps)

    # If we broke neighbor order, do a single forward/backward pass.
    for i in range(1, len(pts)):
        if pts[i] <= pts[i - 1]:
            pts[i] = pts[i - 1] + eps
    for i in range(len(pts) - 2, -1, -1):
        if pts[i] >= pts[i + 1]:
            pts[i] = pts[i + 1] - eps
    return pts


def apply_action_to_state(
    *,
    cfg: Dict[str, object],
    state: Dict[str, Any],
    action: Action,
    base_cfg: Dict[str, object] | None = None,
) -> Dict[str, Any]:
    """Apply an approved action into state['overrides'].

    base_cfg: 개선 override를 적용하기 전 설정(기본값 + 작물 설정). 주어지면
    membership_delta_cap을 '한 번의 변경량'뿐 아니라 '기준점 대비 누적 이동량' 한도로도 적용한다.

    Returns a small dict describing what changed (for logging).
    """
    overrides = state.get("overrides")
    if not isinstance(overrides, dict):
        state["overrides"] = {"memberships": {}, "rule_weights": {}}
        overrides = state["overrides"]

    mem_ov = overrides.get("memberships")
    if not isinstance(mem_ov, dict):
        overrides["memberships"] = {}
        mem_ov = overrides["memberships"]

    rw_ov = overrides.get("rule_weights")
    if not isinstance(rw_ov, dict):
        overrides["rule_weights"] = {}
        rw_ov = overrides["rule_weights"]

    imp = cfg.get("improvement") if isinstance(cfg.get("improvement"), dict) else {}
    delta_caps = (imp or {}).get("membership_delta_cap", {}) if isinstance((imp or {}).get("membership_delta_cap"), dict) else {}
    rw_bounds = (imp or {}).get("rule_weight_bounds", {}) if isinstance((imp or {}).get("rule_weight_bounds"), dict) else {}
    rw_min = float(rw_bounds.get("min", 0.8))
    rw_max = float(rw_bounds.get("max", 1.25))

    if action.action_type == "membership_points":
        if not (action.var_key and action.shape_label is not None and action.point_index is not None and action.delta is not None):
            return {"ok": False, "reason": "invalid membership action"}

        var_key = action.var_key
        # 한도 키: 변수명 그대로(water_temperature, flow_ratio …), EC_water/EC_hydro 는 "EC".
        # (예전 split("_")[0]는 "water_temperature" → "water" 처럼 잘못된 키를 먼저 찾았다)
        cap_key = "EC" if var_key.startswith("EC_") else var_key
        cap = float(delta_caps.get(cap_key, delta_caps.get(var_key, 0.0)) or 0.0)
        d = float(action.delta)
        if cap > 0:
            d = max(min(d, cap), -cap)

        shapes = _get_var_shapes(cfg, var_key)
        if not isinstance(shapes, list):
            return {"ok": False, "reason": f"unknown var_key: {var_key}"}

        base_points = None
        for s in shapes:
            if getattr(s, "label", None) == action.shape_label:
                base_points = list(getattr(s, "points", ()))
                break
        if not base_points:
            return {"ok": False, "reason": f"unknown shape_label: {action.shape_label}"}

        # start from existing override if present
        var_map = mem_ov.get(var_key)
        if not isinstance(var_map, dict):
            var_map = {}
            mem_ov[var_key] = var_map

        cur_pts = var_map.get(action.shape_label)
        if isinstance(cur_pts, list) and all(isinstance(x, (int, float)) for x in cur_pts):
            pts = [float(x) for x in cur_pts]
        else:
            pts = [float(x) for x in base_points]

        idx = int(action.point_index)
        if idx < 0 or idx >= len(pts):
            return {"ok": False, "reason": "point_index out of range"}

        before = pts[idx]
        new_val = float(pts[idx]) + d

        # 누적 한도: 같은 액션이 반복 승인돼도 기준점에서 cap 이상 멀어지지 않게 한다.
        if cap > 0 and base_cfg is not None:
            ref_shapes = _get_var_shapes(base_cfg, var_key)
            ref_pts = None
            if isinstance(ref_shapes, list):
                for s in ref_shapes:
                    if getattr(s, "label", None) == action.shape_label:
                        ref_pts = list(getattr(s, "points", ()))
                        break
            if ref_pts and idx < len(ref_pts):
                ref = float(ref_pts[idx])
                new_val = max(min(new_val, ref + cap), ref - cap)

        if abs(new_val - before) <= 1e-12:
            return {"ok": False, "reason": "membership_delta_cap reached"}

        pts[idx] = new_val
        pts = _clamp_monotonic(pts, idx)
        var_map[action.shape_label] = pts
        return {
            "ok": True,
            "type": "membership_points",
            "var_key": var_key,
            "shape_label": action.shape_label,
            "point_index": idx,
            "before": before,
            "after": pts[idx],
            "delta_applied": pts[idx] - before,
        }

    if action.action_type == "rule_weight":
        if not (action.rule_id and action.multiplier):
            return {"ok": False, "reason": "invalid rule_weight action"}
        rid = action.rule_id
        base_rw = cfg.get("rule_weights") if isinstance(cfg.get("rule_weights"), dict) else {}
        cur = float(rw_ov.get(rid, base_rw.get(rid, 1.0)))
        before = cur
        cur *= float(action.multiplier)
        cur = max(min(cur, rw_max), rw_min)
        rw_ov[rid] = cur
        return {
            "ok": True,
            "type": "rule_weight",
            "rule_id": rid,
            "before": before,
            "after": cur,
            "multiplier": float(action.multiplier),
        }

    return {"ok": False, "reason": "unknown action_type"}
