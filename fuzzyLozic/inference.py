# inference.py
"""
Mamdani fuzzy inference engine.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from membership import evaluate, sample_universe
from rules import Rule, build_rules


def _evaluate_output_set(output_cfg, x: float) -> float:
    return evaluate(output_cfg.shape, output_cfg.points, x)


def _apply_rule_to_output(
    rule: Rule,
    firing_strength: float,
    output_points: List[float],
    output_cfgs,
) -> List[float]:
    """
    Clip the conclusion set and return the clipped membership across universe.
    Implication: min-clipping (Mamdani)
    """
    label = rule.dynamic_label_fn(firing_strength) if rule.dynamic_label_fn else rule.conclusion_label
    cfg = next(cfg for cfg in output_cfgs if cfg.label == label)
    return [min(firing_strength, _evaluate_output_set(cfg, x)) for x in output_points]


def aggregate_outputs(per_label_clipped: Dict[str, List[float]]) -> List[float]:
    """
    Aggregation across all rule outputs: max
    """
    agg = [0.0 for _ in next(iter(per_label_clipped.values()))]
    for vals in per_label_clipped.values():
        agg = [max(a, b) for a, b in zip(agg, vals)]
    return agg


def centroid(universe: List[float], aggregated: List[float]) -> float:
    """
    Defuzzification: centroid
    If area is zero, return 0.0 (caller may override via fallback).
    """
    numerator = sum(x * mu for x, mu in zip(universe, aggregated))
    denominator = sum(aggregated)
    if denominator == 0:
        return 0.0
    return numerator / denominator


def infer(
    memberships: Dict[str, Dict[str, float]],
    cfg: Dict[str, object],
    delta_flags: Dict[str, bool] | None = None,
) -> Tuple[float, str, Dict[str, float], Dict[str, float], List[Dict[str, object]]]:
    """
    Execute Mamdani inference.

    delta_flags is a generic dict for boolean flags. It may include:
    - delta_policy keys (e.g., "T": True) for rapid changes
    - time-series flags (prefix TS_*) for explainable contributions
    """
    rules: Dict[str, Rule] = build_rules(cfg)
    # Optional per-rule weights (improvement module hook). Default is 1.0.
    rule_weights = cfg.get("rule_weights", {}) if isinstance(cfg.get("rule_weights"), dict) else {}
    output_cfgs = cfg["output_risk_sets"]
    universe_cfg = cfg["risk_universe"]
    universe = sample_universe(universe_cfg["min"], universe_cfg["max"], universe_cfg["step"])

    # per output label, store clipped membership across universe
    per_label_clipped: Dict[str, List[float]] = {ocfg.label: [0.0 for _ in universe] for ocfg in output_cfgs}

    fired_rules: List[Dict[str, object]] = []
    warning_rule_policies = cfg["rule_policies"]["R12"]
    warning_strengths: List[float] = []
    delta_flags = delta_flags or {}

    # 1) Evaluate rules (except R12 meta)
    for rule_id, rule in rules.items():
        if rule_id == "R12":
            continue

        strength = rule.evaluate(memberships)
        if rule_id in rule_weights:
            try:
                strength *= float(rule_weights[rule_id])
            except Exception:
                pass
        if strength <= cfg["rule_execution"]["tolerance"]:
            continue

        clipped = _apply_rule_to_output(rule, strength, universe, output_cfgs)
        label = rule.dynamic_label_fn(strength) if rule.dynamic_label_fn else rule.conclusion_label

        per_label_clipped[label] = [max(a, b) for a, b in zip(per_label_clipped[label], clipped)]

        fired_rules.append(
            {
                "rule_id": rule.rule_id,
                "firing_strength": strength,
                "human_readable_reason": rule.human_readable_reason,
                "conclusion_label": label,
            }
        )

        # collect warning-like firings for R12 meta
        if rule.rule_id in warning_rule_policies["warning_rules"] and label in {"경고", "위험"}:
            warning_strengths.append(strength)

    # 2) Optional delta policy: sudden change adds warning weight
    # NOTE: only consume keys configured in delta_policy.thresholds
    delta_cfg = cfg.get("delta_policy", {})
    if delta_cfg.get("enabled"):
        delta_weight = float(delta_cfg.get("weight", 0.0))
        delta_keys = set((delta_cfg.get("thresholds", {}) or {}).keys())
        triggered = [k for k, v in delta_flags.items() if v and (k in delta_keys)]
        if triggered and delta_weight > 0:
            synthetic_rule = Rule(
                rule_id="DELTA",
                premise=None,  # type: ignore[arg-type]
                conclusion_label="경고",
                human_readable_reason=f"급변 감지({','.join(triggered)}) → 경고 가중",
            )
            clipped = _apply_rule_to_output(synthetic_rule, delta_weight, universe, output_cfgs)
            per_label_clipped["경고"] = [max(a, b) for a, b in zip(per_label_clipped["경고"], clipped)]
            fired_rules.append(
                {
                    "rule_id": synthetic_rule.rule_id,
                    "firing_strength": delta_weight,
                    "human_readable_reason": synthetic_rule.human_readable_reason,
                    "conclusion_label": synthetic_rule.conclusion_label,
                }
            )

    # 2.5) Optional time-series policy: qualitative flags add explainable weight
    ts_cfg = cfg.get("timeseries_policy", {})
    if ts_cfg.get("enabled"):
        weights = ts_cfg.get("weights", {}) or {}
        labels = ts_cfg.get("labels", {}) or {}

        def _apply_ts_synthetic(rule_id: str, label: str, reason: str, strength: float) -> None:
            if strength <= cfg["rule_execution"]["tolerance"]:
                return
            synthetic_rule = Rule(
                rule_id=rule_id,
                premise=None,  # type: ignore[arg-type]
                conclusion_label=label,
                human_readable_reason=reason,
            )
            clipped = _apply_rule_to_output(synthetic_rule, strength, universe, output_cfgs)
            per_label_clipped[label] = [max(a, b) for a, b in zip(per_label_clipped[label], clipped)]
            fired_rules.append(
                {
                    "rule_id": synthetic_rule.rule_id,
                    "firing_strength": strength,
                    "human_readable_reason": synthetic_rule.human_readable_reason,
                    "conclusion_label": synthetic_rule.conclusion_label,
                }
            )

        # Trend (FAST adds bonus)
        trend_flags = [k for k, v in delta_flags.items() if v and k.startswith("TS_TREND_")]
        trend_fast = [k for k in trend_flags if k.endswith("_FAST")]
        if trend_flags:
            per = float(weights.get("trend_per_flag", 0.0))
            bonus = float(weights.get("trend_fast_bonus", 0.0))
            mx = float(weights.get("trend_max", 0.0))
            strength = len(trend_flags) * per + len(trend_fast) * bonus
            strength = min(strength, mx) if mx > 0 else strength
            _apply_ts_synthetic(
                "TS_TREND",
                str(labels.get("trend", "경고")),
                f"시계열 추세 감지({','.join(trend_flags)}) → 가중",
                strength,
            )

        # Volatility
        vol_flags = [k for k, v in delta_flags.items() if v and k.startswith("TS_VOL_")]
        if vol_flags:
            per = float(weights.get("vol_per_flag", 0.0))
            mx = float(weights.get("vol_max", 0.0))
            strength = len(vol_flags) * per
            strength = min(strength, mx) if mx > 0 else strength
            _apply_ts_synthetic(
                "TS_VOL",
                str(labels.get("volatility", "주의")),
                f"시계열 변동성 감지({','.join(vol_flags)}) → 가중",
                strength,
            )

        # Persistence
        persist_flags = [k for k, v in delta_flags.items() if v and k.startswith("TS_PERSIST_")]
        if persist_flags:
            per = float(weights.get("persist_per_flag", 0.0))
            mx = float(weights.get("persist_max", 0.0))
            strength = len(persist_flags) * per
            strength = min(strength, mx) if mx > 0 else strength
            _apply_ts_synthetic(
                "TS_PERSIST",
                str(labels.get("persistence", "경고")),
                f"위험 징후 지속 감지({','.join(persist_flags)}) → 가중",
                strength,
            )

        # Spike
        spike_flags = [k for k, v in delta_flags.items() if v and k.startswith("TS_SPIKE_")]
        if spike_flags:
            per = float(weights.get("spike_per_flag", 0.0))
            mx = float(weights.get("spike_max", 0.0))
            strength = len(spike_flags) * per
            strength = min(strength, mx) if mx > 0 else strength
            _apply_ts_synthetic(
                "TS_SPIKE",
                str(labels.get("spike", "경고")),
                f"단기 급변(스파이크) 감지({','.join(spike_flags)}) → 가중",
                strength,
            )

        # Recovery-fail
        rf_flags = [k for k, v in delta_flags.items() if v and k.startswith("TS_RECOVERY_FAIL_")]
        if rf_flags:
            per = float(weights.get("recovery_fail_per_flag", 0.0))
            mx = float(weights.get("recovery_fail_max", 0.0))
            strength = len(rf_flags) * per
            strength = min(strength, mx) if mx > 0 else strength
            _apply_ts_synthetic(
                "TS_RECOVERY_FAIL",
                str(labels.get("recovery_fail", "경고")),
                f"회복 실패 징후 감지({','.join(rf_flags)}) → 가중",
                strength,
            )

    # 3) R12 meta rule: if many warning-level rules fire strongly, promote toward danger
    if len([s for s in warning_strengths if s >= warning_rule_policies["firing_threshold"]]) >= warning_rule_policies[
        "count_threshold"
    ]:
        strength = warning_rule_policies["promote_strength"]
        r12_rule = build_rules(cfg)["R12"]
        clipped = _apply_rule_to_output(r12_rule, strength, universe, output_cfgs)
        per_label_clipped["위험"] = [max(a, b) for a, b in zip(per_label_clipped["위험"], clipped)]
        fired_rules.append(
            {
                "rule_id": "R12",
                "firing_strength": strength,
                "human_readable_reason": r12_rule.human_readable_reason,
                "conclusion_label": "위험",
            }
        )

    # 4) Aggregate and defuzzify
    aggregated = aggregate_outputs(per_label_clipped)
    risk_score = centroid(universe, aggregated)

    # peaks per label (aggregated output strength insight)
    label_peaks = {ocfg.label: max(per_label_clipped[ocfg.label]) for ocfg in output_cfgs}

    # score membership per label (for label decision)
    score_membership = {ocfg.label: evaluate(ocfg.shape, ocfg.points, risk_score) for ocfg in output_cfgs}

    # 5) Fallback: if nothing fired (or aggregated is empty), avoid misleading output
    max_peak = max(label_peaks.values()) if label_peaks else 0.0
    if (not fired_rules) and max_peak <= cfg["rule_execution"]["tolerance"]:
        fallback_label = cfg["rule_execution"].get("fallback_label", "경계")
        fallback_score = float(cfg["rule_execution"].get("fallback_score", 50.0))
        risk_score = fallback_score
        score_membership = {ocfg.label: evaluate(ocfg.shape, ocfg.points, risk_score) for ocfg in output_cfgs}
        return risk_score, fallback_label, score_membership, label_peaks, fired_rules

    # 6) Label decision: pick label with max membership at the score; ties → higher risk
    label_order = {label: idx for idx, label in enumerate(["매우안정", "안정", "경계", "주의", "경고", "위험"])}
    best_label = max(score_membership.items(), key=lambda kv: (kv[1], label_order[kv[0]]))[0]

    # 7) Conservative policy: promote to "위험" if danger signals are strong enough
    danger_score_threshold = 70.0
    danger_peak_threshold = 0.6
    danger_rule_threshold = 0.6

    danger_peak = float(label_peaks.get("위험", 0.0))
    max_danger_rule = 0.0
    for r in fired_rules:
        if r.get("conclusion_label") == "위험":
            max_danger_rule = max(max_danger_rule, float(r.get("firing_strength", 0.0)))

    promote_to_danger = (
        (risk_score >= danger_score_threshold and danger_peak >= danger_peak_threshold)
        or (max_danger_rule >= danger_rule_threshold)
    )

    if promote_to_danger and best_label in {"경계", "주의", "경고"}:
        best_label = "위험"

    return risk_score, best_label, score_membership, label_peaks, fired_rules
