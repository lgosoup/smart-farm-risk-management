"""
Rule base definitions for the fuzzy inference system.
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
    rules["R1"] = Rule(
        "R1",
        atom("DO", "치명적 낮음"),
        "위험",
        "DO 치명적 낮음 → 위험",
    )
    rules["R2"] = Rule(
        "R2",
        atom("T", "위험(급고온)"),
        "위험",
        "수온 급고온 → 위험",
    )
    rules["R3"] = Rule(
        "R3",
        atom("FR", "정체(위험)"),
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
        atom("Turbidity", "극도로 탁함(위험)"),
        "위험",
        "탁도 극도로 탁함 → 위험",
    )

    # B. 복합 상호작용
    rules["R6"] = Rule(
        "R6",
        all_of(
            atom("T", "경고(고온)"),
            any_of(atom("DO", "낮음"), atom("DO", "매우낮음"), atom("DO", "치명적 낮음")),
        ),
        "위험",
        "수온 고온 + DO 저하(낮음/매우낮음/치명) → 위험",
    )
    rules["R7"] = Rule(
        "R7",
        all_of(
            any_of(atom("FR", "낮음(주의)"), atom("FR", "매우낮음(경고)"), atom("FR", "정체(위험)")),
            any_of(atom("DO", "낮음"), atom("DO", "매우낮음"), atom("DO", "치명적 낮음")),
        ),
        "위험",
        "유량 저하(낮음/매우낮음/정체) + DO 저하(낮음/매우낮음/치명) → 위험",
    )
    rules["R8"] = Rule(
        "R8",
        all_of(
            any_of(atom("FR", "낮음(주의)"), atom("FR", "매우낮음(경고)"), atom("FR", "정체(위험)")),
            any_of(
                atom("Turbidity", "탁함(주의)"),
                atom("Turbidity", "매우 탁함(경고)"),
                atom("Turbidity", "극도로 탁함(위험)"),
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
        all_of(atom("EC", "매우높음(위험)"), any_of(atom("DO", "경계"), atom("DO", "낮음"), atom("DO", "치명적 낮음"))),
        rule_policies["R9"]["base_label"],
        "EC 매우높음 & DO 경계 이하 → 경고/위험",
        dynamic_label_fn=r9_label,
    )
    rules["R10"] = Rule(
        "R10",
        all_of(atom("T", "경계(상승)"), atom("DO", "경계")),
        "경고",
        "수온 경계 & DO 경계 → 경고",
    )

    # A/B 보완: 단일 경고 규칙(규칙 0발화 방지 + 설명 가능성 향상)
    rules["R15"] = Rule("R15", atom("DO", "매우낮음"), "경고", "용존산소가 매우낮음 → 경고")
    rules["R16"] = Rule("R16", atom("T", "경고(고온)"), "경고", "수온이 고온 구간 → 경고")
    rules["R17"] = Rule("R17", atom("FR", "매우낮음(경고)"), "경고", "유량이 매우낮음 → 경고")
    rules["R18"] = Rule("R18", atom("Turbidity", "매우 탁함(경고)"), "경고", "탁도가 매우 탁함 → 경고")
    rules["R19"] = Rule("R19", atom("EC", "높음(경고)"), "경고", "EC가 높음 구간 → 경고")
    rules["R20"] = Rule(
        "R20",
        any_of(atom("pH", "약산성(경고–주의)"), atom("pH", "약알칼리(경고–주의)")),
        "주의",
        "pH가 최적 범위를 벗어남(약산성/약알칼리) → 주의",
    )

    # C. 누적 승급
    rules["R11"] = Rule(
        "R11",
        all_of(atom("T", "경계(상승)"), atom("DO", "경계"), atom("FR", "경계")),
        "경고",
        "수온/DO/유량이 모두 경계 → 경고로 승급",
    )
    # R12 handled in inference via policy; stub rule for reference.
    rules["R12"] = Rule(
        "R12",
        atom("DO", "경계"),
        "위험",
        "경고급 규칙 다발 → 위험으로 승급",
    )

    # D. 안정 확인
    rules["R13"] = Rule(
        "R13",
        all_of(
            atom("T", "안정(최적)"),
            atom("DO", "안정(목표)"),
            atom("pH", "안정(최적)"),
            atom("FR", "안정"),
            atom("EC", "안정"),
            atom("Turbidity", "맑음(안정)"),
        ),
        "매우안정",
        "모든 핵심 지표가 안정 → 매우안정",
    )
    rules["R14"] = Rule(
        "R14",
        all_of(
            atom("T", "안정(최적)"),
            atom("DO", "안정(목표)"),
            atom("pH", "안정(최적)"),
            atom("FR", "안정"),
        ),
        "안정",
        "핵심 4개가 안정 → 안정",
    )
    return rules
