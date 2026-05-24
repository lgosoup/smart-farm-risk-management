# fuzzify.py
"""
Input preprocessing and fuzzification.

Policy:
- Sensor inputs MUST NOT be modified (no smoothing, no clipping, no imputation).
- Time-series behavior is handled via qualitative flags computed from stored history.
"""
from __future__ import annotations

import statistics
from typing import Dict, List, Optional

from config import VariableConfig, load_config
from membership import evaluate


def load_default_config() -> Dict[str, object]:
    # If improvement_state.json exists, it can override memberships/rule weights.
    return load_config()


def preprocess_inputs(
    raw_inputs: Dict[str, Optional[float]],
    cfg: Dict[str, object],
    history: Optional[List[Dict[str, Optional[float]]]] = None,
) -> Dict[str, Optional[float]]:
    """
    IMPORTANT POLICY:
    Do NOT adjust/modify sensor inputs.

    This function is kept for compatibility but returns inputs as-is.
    """
    return dict(raw_inputs)


def _fuzzify_single(value: Optional[float], var_cfg: VariableConfig) -> Dict[str, float]:
    memberships: Dict[str, float] = {}
    for shape in var_cfg.shapes:
        memberships[shape.label] = evaluate(shape.shape, shape.points, value) if value is not None else 0.0
    return memberships


def fuzzify_inputs(
    inputs: Dict[str, Optional[float]],
    cfg: Dict[str, object],
    ec_mode: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Convert crisp inputs to membership degrees per variable.
    """
    variables = cfg["variables"]
    mode = ec_mode or cfg.get("ec_mode_default", "water")

    memberships: Dict[str, Dict[str, float]] = {}
    for var_name, var_cfg in variables.items():
        if var_name == "EC":
            var_cfg = var_cfg[mode]
        memberships[var_name] = _fuzzify_single(inputs.get(var_name), var_cfg)

    return memberships


def compute_delta_flags(
    current_inputs: Dict[str, Optional[float]],
    previous_inputs: Optional[Dict[str, Optional[float]]],
    cfg: Dict[str, object],
) -> Dict[str, bool]:
    """
    Compare current vs previous to detect rapid changes (optional feature).
    """
    delta_cfg = cfg.get("delta_policy", {})
    if not delta_cfg.get("enabled") or not previous_inputs:
        return {}

    thresholds = delta_cfg.get("thresholds", {})
    flags: Dict[str, bool] = {}
    for key, th in thresholds.items():
        cur = current_inputs.get(key)
        prev = previous_inputs.get(key)
        if cur is None or prev is None:
            flags[key] = False
        else:
            flags[key] = abs(float(cur) - float(prev)) >= float(th)
    return flags


def compute_integrity_flags(
    current_inputs: Dict[str, Optional[float]],
    previous_inputs: Optional[Dict[str, Optional[float]]],
    cfg: Dict[str, object],
) -> Dict[str, object]:
    """
    Sensor integrity checks (flags only; NEVER modifies inputs).

    Returns:
      {
        "out_of_range": {var: True, ...},
        "spike": {var: {"direction": "up|down", "delta": float}, ...}
      }
    """
    ip = cfg.get("integrity_policy", {})
    if not ip.get("enabled"):
        return {"out_of_range": {}, "spike": {}}

    ranges = ip.get("ranges", {}) or {}
    spike_th = ip.get("spike_thresholds", {}) or {}

    out_of_range: Dict[str, bool] = {}
    spike: Dict[str, Dict[str, object]] = {}

    for var, (lo, hi) in ranges.items():
        v = current_inputs.get(var)
        if v is None:
            continue
        if float(v) < float(lo) or float(v) > float(hi):
            out_of_range[var] = True

    if previous_inputs:
        for var, th in spike_th.items():
            cur = current_inputs.get(var)
            prev = previous_inputs.get(var)
            if cur is None or prev is None:
                continue
            delta = float(cur) - float(prev)
            if abs(delta) >= float(th):
                spike[var] = {"direction": "up" if delta > 0 else "down", "delta": delta}

    return {"out_of_range": out_of_range, "spike": spike}


def compute_timeseries_flags(
    current_inputs: Dict[str, Optional[float]],
    history: Optional[List[Dict[str, Optional[float]]]],
    cfg: Dict[str, object],
) -> Dict[str, bool]:
    """
    Derive qualitative time-series flags (trend/volatility/persistence/spike/recovery-fail).

    Returns a dict of {flag_name: True} for triggered flags only.
    """
    ts_cfg = cfg.get("timeseries_policy", {})
    if not ts_cfg.get("enabled") or not history:
        return {}

    window = int(ts_cfg.get("window", 5))
    if window <= 1:
        return {}

    # history contains previous samples; we use last (window-1) plus current to make window-sized series
    recent = history[-(window - 1) :] if window else history
    flags: Dict[str, bool] = {}

    # helpers
    def _series(var: str) -> List[float]:
        s = [row.get(var) for row in recent if row.get(var) is not None]
        cur = current_inputs.get(var)
        if cur is not None:
            s.append(cur)
        return [float(x) for x in s if x is not None]

    # ---------- Trend flags (with FAST level) ----------
    trend_cfg = ts_cfg.get("trend", {}) or {}
    for var, spec in trend_cfg.items():
        direction = str(spec.get("direction", ""))
        threshold = float(spec.get("threshold", 0.0))
        fast_threshold = float(spec.get("fast_threshold", 0.0)) if spec.get("fast_threshold") is not None else 0.0

        s = _series(var)
        if len(s) < 2 or threshold <= 0:
            continue

        change = s[-1] - s[0]
        if direction == "up":
            if change >= threshold:
                flags[f"TS_TREND_{var}_UP"] = True
            if fast_threshold > 0 and change >= fast_threshold:
                flags[f"TS_TREND_{var}_UP_FAST"] = True
        elif direction == "down":
            if change <= -threshold:
                flags[f"TS_TREND_{var}_DOWN"] = True
            if fast_threshold > 0 and change <= -fast_threshold:
                flags[f"TS_TREND_{var}_DOWN_FAST"] = True

    # ---------- Volatility flags ----------
    vol_cfg = ts_cfg.get("volatility", {}) or {}
    for var, threshold in vol_cfg.items():
        try:
            th = float(threshold)
        except (TypeError, ValueError):
            continue
        if th <= 0:
            continue

        s = _series(var)
        if len(s) < 3:
            continue

        if float(statistics.pstdev(s)) >= th:
            flags[f"TS_VOL_{var}"] = True

    # ---------- Persistence flags ----------
    persist_cfg = ts_cfg.get("persistence", {}) or {}
    for var, spec in persist_cfg.items():
        op = str(spec.get("op", ""))
        try:
            th = float(spec.get("threshold", 0.0))
        except (TypeError, ValueError):
            continue
        count = int(spec.get("count", 0))
        if count <= 0:
            continue

        s = _series(var)
        if len(s) < count:
            continue
        last_vals = s[-count:]

        if op == "ge":
            ok = all(x >= th for x in last_vals)
        elif op == "le":
            ok = all(x <= th for x in last_vals)
        else:
            ok = False

        if ok:
            direction = "HIGH" if op == "ge" else "LOW"
            flags[f"TS_PERSIST_{var}_{direction}"] = True

    # ---------- Spike flags (single-step) ----------
    spike_cfg = ts_cfg.get("spike", {}) or {}
    if recent:
        prev_sample = recent[-1]
        for var, spec in spike_cfg.items():
            cur = current_inputs.get(var)
            prev = prev_sample.get(var)
            if cur is None or prev is None:
                continue
            delta = float(cur) - float(prev)
            up_th = float(spec.get("up", 0.0))
            down_th = float(spec.get("down", 0.0))
            if up_th > 0 and delta >= up_th:
                flags[f"TS_SPIKE_{var}_UP"] = True
            if down_th > 0 and delta <= -down_th:
                flags[f"TS_SPIKE_{var}_DOWN"] = True

    # ---------- Recovery-fail flags ----------
    rf_cfg = ts_cfg.get("recovery_fail", {}) or {}
    for var, spec in rf_cfg.items():
        bad_op = str(spec.get("bad_op", ""))
        bad_threshold = float(spec.get("bad_threshold", 0.0))
        count = int(spec.get("count", 0))
        min_recover_delta = float(spec.get("min_recover_delta", 0.0))
        if count <= 1:
            continue

        s = _series(var)
        if len(s) < count:
            continue
        last_vals = s[-count:]

        if bad_op == "le":
            in_bad = all(x <= bad_threshold for x in last_vals)
            # recovering means last - first >= min_recover_delta
            recovering = (last_vals[-1] - last_vals[0]) >= min_recover_delta
        elif bad_op == "ge":
            in_bad = all(x >= bad_threshold for x in last_vals)
            recovering = (last_vals[0] - last_vals[-1]) >= min_recover_delta
        else:
            in_bad = False
            recovering = False

        if in_bad and not recovering:
            flags[f"TS_RECOVERY_FAIL_{var}"] = True

    return flags
