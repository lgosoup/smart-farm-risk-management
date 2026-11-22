"""
Rule base definitions for the fuzzy inference system.

Sensor variables used:
  water_temperature, pH, EC, flow_ratio, turbidity
  (air_temperature, humidity: monitoring only — no rules)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


@dataclass
class Rule:
    rule_id: str
    premise: Any
    conclusion_label: str
    human_readable_reason: str
    dynamic_label_fn: Optional[Callable[[float], str]] = None

    def evaluate(self, memberships: Dict[str, Dict[str, float]]) -> float:
        return float(self.premise(memberships))


def atom(var: str, label: str):
    def _eval(memberships: Dict[str, Dict[str, float]]) -> float:
        return float(memberships.get(var, {}).get(label, 0.0))

    return _eval


def all_of(*conds):
    def _eval(memberships: Dict[str, Dict[str, float]]) -> float:
        if not conds:
            return 0.0
        return float(min(c(memberships) for c in conds))

    return _eval


def any_of(*conds):
    def _eval(memberships: Dict[str, Dict[str, float]]) -> float:
        if not conds:
            return 0.0
        return float(max(c(memberships) for c in conds))

    return _eval


def build_rules(cfg: Dict[str, object]) -> Dict[str, Rule]:
    """
    Build the rule base. Labels must match those in config.py.
    """
    rule_policies = cfg["rule_policies"]
    rules: Dict[str, Rule] = {}

    # A. 단독 치명
    rules["R2"] = Rule(
        "R2",
        atom("water_temperature", "위험(급고온)"),
        "위험",
        "수온 급고온 → 위험",
    )
    rules["R3"] = Rule(
        "R3",
        atom("flow_ratio", "정체(위험)"),
        "위험",
        "유량 정체 → 위험",
    )

    def r4_label(strength: float) -> str:
        return (
            rule_policies["R4"]["promote_label"]
            if strength >= rule_policies["R4"]["promote_threshold"]
            else rule_policies["R4"]["base_label"]
        )

    rules["R4"] = Rule(
        "R4",
        any_of(atom("pH", "강산성(위험)"), atom("pH", "강알칼리(위험)")),
        rule_policies["R4"]["base_label"],
        "pH 극단(강산성/강알칼리) → 경고/위험",
        dynamic_label_fn=r4_label,
    )
    rules["R5"] = Rule(
        "R5",
        atom("turbidity", "극도로 탁함(위험)"),
        "위험",
        "탁도 극도로 탁함 → 위험",
    )

    # B. 복합 상호작용
    rules["R6"] = Rule(
        "R6",
        all_of(
            atom("water_temperature", "경고(고온)"),
            any_of(
                atom("flow_ratio", "낮음(주의)"),
                atom("flow_ratio", "매우낮음(경고)"),
                atom("flow_ratio", "정체(위험)"),
            ),
        ),
        "위험",
        "수온 고온 + 유량 저하(낮음/매우낮음/정체) → 위험",
    )
    rules["R7"] = Rule(
        "R7",
        all_of(
            atom("EC", "매우높음(위험)"),
            any_of(
                atom("flow_ratio", "낮음(주의)"),
                atom("flow_ratio", "매우낮음(경고)"),
                atom("flow_ratio", "정체(위험)"),
            ),
        ),
        "위험",
        "EC 매우높음 + 유량 저하(낮음/매우낮음/정체) → 위험",
    )
    rules["R8"] = Rule(
        "R8",
        all_of(
            any_of(atom("flow_ratio", "낮음(주의)"), atom("flow_ratio", "매우낮음(경고)"), atom("flow_ratio", "정체(위험)")),
            any_of(
                atom("turbidity", "탁함(주의)"),
                atom("turbidity", "매우 탁함(경고)"),
                atom("turbidity", "극도로 탁함(위험)"),
            ),
        ),
        "경고",
        "유량 저하 + 탁도 악화(탁함/매우 탁함/극도로 탁함) → 경고",
    )

    def r9_label(strength: float) -> str:
        return (
            rule_policies["R9"]["promote_label"]
            if strength >= rule_policies["R9"]["promote_threshold"]
            else rule_policies["R9"]["base_label"]
        )

    rules["R9"] = Rule(
        "R9",
        all_of(
            atom("EC", "매우높음(위험)"),
            atom("flow_ratio", "경계"),
        ),
        rule_policies["R9"]["base_label"],
        "EC 매우높음 & 유량 경계 → 경고/위험",
        dynamic_label_fn=r9_label,
    )
    rules["R10"] = Rule(
        "R10",
        all_of(atom("water_temperature", "경계(상승)"), atom("flow_ratio", "경계")),
        "경고",
        "수온 경계 & 유량 경계 → 경고",
    )

    # A/B 보완: 단일 경고 규칙
    rules["R16"] = Rule("R16", atom("water_temperature", "경고(고온)"), "경고", "수온이 고온 구간 → 경고")
    rules["R17"] = Rule("R17", atom("flow_ratio", "매우낮음(경고)"), "경고", "유량이 매우낮음 → 경고")
    rules["R18"] = Rule("R18", atom("turbidity", "매우 탁함(경고)"), "경고", "탁도가 매우 탁함 → 경고")
    rules["R19"] = Rule("R19", atom("EC", "높음(경고)"), "경고", "EC가 높음 구간 → 경고")
    rules["R21"] = Rule("R21", atom("EC", "매우높음(위험)"), "위험", "EC 매우높음(염류 스트레스) → 위험")
    rules["R20"] = Rule(
        "R20",
        any_of(atom("pH", "약산성(경고–주의)"), atom("pH", "약알칼리(경고–주의)")),
        "주의",
        "pH가 최적 범위를 벗어남(약산성/약알칼리) → 주의",
    )

    # E. 단일 경계·주의 규칙: 한 변수만 경계/주의 구간에 있을 때 발화하는 규칙이 없으면
    #    fallback(50점)으로 점프하므로, 위험도가 연속적으로 변하도록 빈 구간을 채운다.
    rules["R22"] = Rule("R22", atom("water_temperature", "경계(상승)"), "경계", "수온이 경계(상승) 구간 → 경계")
    rules["R23"] = Rule("R23", atom("water_temperature", "낮음"), "경계", "수온이 낮음 구간 → 경계")
    rules["R24"] = Rule("R24", atom("water_temperature", "매우낮음"), "주의", "수온이 매우낮음 구간 → 주의")
    rules["R25"] = Rule("R25", atom("flow_ratio", "경계"), "경계", "유량이 경계 구간 → 경계")
    rules["R26"] = Rule("R26", atom("flow_ratio", "낮음(주의)"), "주의", "유량이 낮음 구간 → 주의")
    rules["R27"] = Rule("R27", atom("flow_ratio", "과다(주의)"), "주의", "유량이 과다 → 주의")
    rules["R28"] = Rule("R28", atom("turbidity", "탁함(주의)"), "주의", "탁도가 탁함 구간 → 주의")
    rules["R29"] = Rule("R29", atom("EC", "낮음"), "경계", "EC가 낮음 구간 → 경계")
    rules["R30"] = Rule("R30", atom("EC", "매우낮음"), "주의", "EC가 매우낮음 구간 → 주의")

    # C. 누적 승급
    rules["R11"] = Rule(
        "R11",
        all_of(atom("water_temperature", "경계(상승)"), atom("flow_ratio", "경계")),
        "경고",
        "수온/유량이 모두 경계 → 경고로 승급",
    )
    # R12 handled in inference via policy; stub rule for reference.
    # 전제는 inference.py의 메타 로직(경고급 규칙 다발)이 대신하므로 여기서는 항상 0.
    # (이전 stub은 'turbidity 약간 탁함 → 위험'으로 읽혀, 직접 evaluate하면 탁도만으로 위험이 나왔다)
    rules["R12"] = Rule(
        "R12",
        lambda memberships: 0.0,
        "위험",
        "경고급 규칙 다발 → 위험으로 승급",
    )

    # D. 안정 확인
    rules["R13"] = Rule(
        "R13",
        all_of(
            atom("water_temperature", "안정(최적)"),
            atom("pH", "안정(최적)"),
            atom("flow_ratio", "안정"),
            atom("EC", "안정"),
            atom("turbidity", "맑음(안정)"),
        ),
        "매우안정",
        "모든 핵심 지표가 안정 → 매우안정",
    )
    rules["R14"] = Rule(
        "R14",
        all_of(
            atom("water_temperature", "안정(최적)"),
            atom("pH", "안정(최적)"),
            atom("flow_ratio", "안정"),
        ),
        "안정",
        "핵심 3개가 안정 → 안정",
    )
    return rules
