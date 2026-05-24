# config.py
"""
Configuration for fuzzy membership functions, output sets, and runtime policies.

Policy note:
- Sensor inputs MUST NOT be modified (no smoothing, no clipping, no imputation).
- Time-series behavior is handled via qualitative flags (trend/volatility/persistence/spike/recovery),
  which can be used as explainable contributors in inference and reporting.

Sensor variables:
  water_temperature : 수온 (°C) — 양액/수조 수온
  pH               : 산도 (0–14)
  EC               : 전기전도도 (mS/cm or dS/m)
  flow_ratio       : 유량비 (무차원, 1.0 = 기준)
  turbidity        : 탁도 (NTU)
  air_temperature  : 공기 온도 °C (SHTC3, 모니터링 전용)
  humidity         : 공기 상대습도 % (SHTC3, 모니터링 전용)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import json
import copy
from pathlib import Path


_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_STATE_PATH = str(_DATA_DIR / "improvement_state.json")
CROP_CONFIG_PATH = str(_DATA_DIR / "crop_config.json")


@dataclass(frozen=True)
class MembershipShape:
    label: str
    shape: str  # "tri" or "trap"
    points: Tuple[float, ...]


@dataclass(frozen=True)
class VariableConfig:
    name: str
    shapes: List[MembershipShape]


@dataclass(frozen=True)
class OutputSetConfig:
    label: str
    shape: str
    points: Tuple[float, ...]


def default_config() -> Dict[str, object]:
    """
    Default config builder. Returns new objects each time to avoid accidental mutation.
    """
    # Water temperature anchors (°C): optimum 12–13, acceptable 10–15
    water_temperature_shapes = [
        MembershipShape("매우낮음", "trap", (-5.0, -1.0, 6.0, 8.0)),
        MembershipShape("낮음", "tri", (7.0, 9.0, 11.0)),
        MembershipShape("안정(최적)", "trap", (10.0, 12.0, 13.0, 15.0)),
        MembershipShape("경계(상승)", "tri", (14.0, 16.0, 18.0)),
        MembershipShape("경고(고온)", "tri", (17.0, 20.0, 24.0)),
        MembershipShape("위험(급고온)", "trap", (22.0, 26.0, 35.0, 40.0)),
    ]

    # pH anchors: optimum 6.0–7.0
    ph_shapes = [
        MembershipShape("강산성(위험)", "trap", (0.0, 0.0, 4.5, 5.5)),
        MembershipShape("약산성(경고–주의)", "tri", (5.2, 5.8, 6.2)),
        MembershipShape("안정(최적)", "trap", (6.0, 6.3, 6.7, 7.0)),
        MembershipShape("약알칼리(경고–주의)", "tri", (6.8, 7.4, 7.8)),
        MembershipShape("강알칼리(위험)", "trap", (7.6, 8.2, 14.0, 14.5)),
    ]

    # Flow ratio FR (unitless, relative)
    flow_ratio_shapes = [
        MembershipShape("정체(위험)", "trap", (0.0, 0.0, 0.35, 0.45)),
        MembershipShape("매우낮음(경고)", "tri", (0.3, 0.45, 0.6)),
        MembershipShape("낮음(주의)", "tri", (0.5, 0.65, 0.8)),
        MembershipShape("경계", "tri", (0.75, 0.85, 0.95)),
        MembershipShape("안정", "trap", (0.9, 1.0, 1.05, 1.1)),
        MembershipShape("과다(주의)", "trap", (1.05, 1.2, 1.6, 2.0)),
    ]

    # EC is mode dependent
    ec_shapes_water = [
        MembershipShape("매우낮음", "trap", (0.0, 0.0, 0.015, 0.03)),
        MembershipShape("낮음", "tri", (0.02, 0.05, 0.1)),
        MembershipShape("안정", "trap", (0.03, 0.08, 0.2, 0.25)),
        MembershipShape("높음(경고)", "tri", (0.18, 0.3, 0.45)),
        MembershipShape("매우높음(위험)", "trap", (0.4, 0.6, 1.0, 1.5)),
    ]
    ec_shapes_hydro = [
        MembershipShape("매우낮음", "trap", (0.0, 0.0, 0.25, 0.45)),
        MembershipShape("낮음", "tri", (0.4, 0.65, 0.9)),
        MembershipShape("안정", "trap", (0.5, 1.0, 2.0, 2.5)),
        MembershipShape("높음(경고)", "tri", (1.8, 2.6, 3.4)),
        MembershipShape("매우높음(위험)", "trap", (3.0, 4.0, 5.5, 6.5)),
    ]

    # Turbidity (NTU) – tunable
    turbidity_shapes = [
        MembershipShape("맑음(안정)", "trap", (0.0, 0.0, 0.5, 1.0)),
        MembershipShape("약간 탁함(경계)", "tri", (0.8, 1.5, 3.0)),
        MembershipShape("탁함(주의)", "tri", (2.5, 5.0, 8.0)),
        MembershipShape("매우 탁함(경고)", "tri", (7.0, 10.0, 14.0)),
        MembershipShape("극도로 탁함(위험)", "trap", (12.0, 15.0, 25.0, 35.0)),
    ]

    # Air temperature (°C) — SHTC3 기반, 모니터링 전용 (퍼지 규칙 미적용)
    air_temperature_shapes = [
        MembershipShape("매우낮음", "trap", (-5.0, 0.0, 10.0, 15.0)),
        MembershipShape("낮음", "tri", (12.0, 15.0, 20.0)),
        MembershipShape("안정(적정)", "trap", (18.0, 22.0, 28.0, 32.0)),
        MembershipShape("높음(주의)", "tri", (30.0, 35.0, 40.0)),
        MembershipShape("매우높음(위험)", "trap", (38.0, 42.0, 60.0, 65.0)),
    ]

    # Humidity (%) — SHTC3 기반, 모니터링 전용 (퍼지 규칙 미적용)
    humidity_shapes = [
        MembershipShape("매우낮음", "trap", (0.0, 0.0, 20.0, 35.0)),
        MembershipShape("낮음", "tri", (25.0, 40.0, 55.0)),
        MembershipShape("적정", "trap", (50.0, 60.0, 75.0, 80.0)),
        MembershipShape("높음(주의)", "tri", (75.0, 85.0, 90.0)),
        MembershipShape("매우높음(위험)", "trap", (88.0, 95.0, 100.0, 100.0)),
    ]

    # Output risk fuzzy sets over 0–100
    risk_outputs = [
        OutputSetConfig("매우안정", "trap", (0.0, 0.0, 10.0, 20.0)),
        OutputSetConfig("안정", "tri", (10.0, 25.0, 35.0)),
        OutputSetConfig("경계", "tri", (25.0, 40.0, 50.0)),
        OutputSetConfig("주의", "tri", (40.0, 55.0, 65.0)),
        OutputSetConfig("경고", "tri", (55.0, 70.0, 80.0)),
        OutputSetConfig("위험", "trap", (70.0, 85.0, 100.0, 100.0)),
    ]

    cfg: Dict[str, object] = {
        "variables": {
            "water_temperature": VariableConfig("water_temperature", water_temperature_shapes),
            "pH": VariableConfig("pH", ph_shapes),
            "flow_ratio": VariableConfig("flow_ratio", flow_ratio_shapes),
            "EC": {
                "water": VariableConfig("EC_water", ec_shapes_water),
                "hydro": VariableConfig("EC_hydro", ec_shapes_hydro),
            },
            "turbidity": VariableConfig("turbidity", turbidity_shapes),
            # air_temperature, humidity: 모니터링 전용 (규칙 미적용)
            "air_temperature": VariableConfig("air_temperature", air_temperature_shapes),
            "humidity": VariableConfig("humidity", humidity_shapes),
        },
        "output_risk_sets": risk_outputs,
        "risk_universe": {"min": 0.0, "max": 100.0, "step": 1.0},
        "ec_mode_default": "water",

        # Input handling policy: DO NOT MODIFY inputs.
        "flow": {
            "accept_raw_flow": False,
            "default_F0": 1.0,
            "treat_excess_as_warning": True,
        },
        "missing_policy": {"strategy": "skip", "default_missing_value": None},
        "outlier_policy": {"clip": False, "ranges": {}},
        "smoothing": {"enabled": False, "window": 0, "fields": []},

        # Sensor integrity checks (flags only; no clipping)
        "integrity_policy": {
            "enabled": True,
            "ranges": {
                "water_temperature": (-5.0, 45.0),
                "pH": (0.0, 14.0),
                "flow_ratio": (0.0, 3.0),
                "EC": (0.0, 10.0),
                "turbidity": (0.0, 100.0),
                "air_temperature": (-10.0, 60.0),
                "humidity": (0.0, 100.0),
            },
            "spike_thresholds": {
                "water_temperature": 5.0,
                "pH": 1.5,
                "flow_ratio": 0.8,
                "EC": 2.5,
                "turbidity": 25.0,
                "air_temperature": 8.0,
                "humidity": 20.0,
            },
        },

        # Rapid change (optional)
        "delta_policy": {
            "enabled": False,
            "thresholds": {"water_temperature": 2.0, "flow_ratio": 0.3},
            "weight": 0.1,
        },

        # Time-series policy (flags only; no input modification)
        "timeseries_policy": {
            "enabled": True,
            "window": 5,
            "trend": {
                "water_temperature": {"direction": "up", "threshold": 1.0, "fast_threshold": 2.5},
                "flow_ratio": {"direction": "down", "threshold": 0.15, "fast_threshold": 0.35},
            },
            "spike": {
                "water_temperature": {"up": 1.5, "down": 1.5},
                "flow_ratio": {"up": 0.2, "down": 0.2},
            },
            "volatility": {"water_temperature": 0.6, "flow_ratio": 0.12},
            "persistence": {
                "water_temperature": {"op": "ge", "threshold": 17.0, "count": 3},
                "flow_ratio": {"op": "le", "threshold": 0.65, "count": 3},
            },
            "recovery_fail": {
                "flow_ratio": {"bad_op": "le", "bad_threshold": 0.65, "count": 3, "min_recover_delta": 0.05},
            },
            "weights": {
                "trend_per_flag": 0.08,
                "trend_fast_bonus": 0.06,
                "trend_max": 0.30,

                "persist_per_flag": 0.12,
                "persist_max": 0.36,

                "vol_per_flag": 0.05,
                "vol_max": 0.15,

                "spike_per_flag": 0.10,
                "spike_max": 0.20,

                "recovery_fail_per_flag": 0.12,
                "recovery_fail_max": 0.24,
            },
            "labels": {
                "trend": "경고",
                "persistence": "경고",
                "volatility": "주의",
                "spike": "경고",
                "recovery_fail": "경고",
            },
        },

        # Alert escalation policy (post inference; for action_key only)
        "alert_policy": {
            "enabled": True,
            "danger_escalation": {"label": "위험", "count": 3, "escalated_action_key": "긴급"},
        },

        # Logging policy (JSONL retention)
        "logging_policy": {
            "inputs_max_records": 2000,
            "packets_max_records": 2000,
        },

        "rule_policies": {
            "R4": {"base_label": "경고", "promote_label": "위험", "promote_threshold": 0.75},
            "R9": {"base_label": "경고", "promote_label": "위험", "promote_threshold": 0.8},
            "R12": {
                "warning_rules": [
                    "R2","R4","R5","R6","R7","R8","R10","R11","R16","R17","R18","R19"
                ],
                "firing_threshold": 0.6,
                "count_threshold": 2,
                "promote_strength": 0.7,
            },
        },
        "rule_execution": {
            "top_n": 5,
            "tolerance": 1e-6,
            "fallback_label": "경계",
            "fallback_score": 50.0,
        },

        # -------- Improvement hooks (bandit + optional LLM gate) --------
        "improvement": {
            "enabled": False,
            "state_path": DEFAULT_STATE_PATH,
            "membership_delta_cap": {
                "water_temperature": 0.5,
                "pH": 0.15,
                "flow_ratio": 0.08,
                "EC": 0.25,
                "turbidity": 1.5,
            },
            "rule_weight_bounds": {
                "min": 0.80,
                "max": 1.25,
            },
        },

        "rule_weights": {
            "R1": 1.0, "R2": 1.0, "R3": 1.0, "R4": 1.0, "R5": 1.0,
            "R6": 1.0, "R7": 1.0, "R8": 1.0, "R9": 1.0, "R10": 1.0,
            "R11": 1.0, "R12": 1.0,
            "R13": 1.0, "R14": 1.0,
            "R16": 1.0, "R17": 1.0, "R18": 1.0, "R19": 1.0, "R20": 1.0,
        },
    }

    return cfg


def _apply_membership_overrides(cfg: Dict[str, object], overrides: Dict[str, object]) -> None:
    """
    Apply membership point overrides in-place.

    Expected override format:
    {
      "memberships": {
        "water_temperature": {"안정(최적)": [10.0, 12.1, 13.0, 15.0], ...},
        "flow_ratio": {...},
        "EC_water": {...},
        "EC_hydro": {...},
      }
    }
    """
    mem = overrides.get("memberships")
    if not isinstance(mem, dict):
        return

    variables = cfg.get("variables")
    if not isinstance(variables, dict):
        return

    def _rebuilt_var(var_key: str, var_cfg_obj) -> object:
        ov = mem.get(var_key)
        if not isinstance(ov, dict):
            return var_cfg_obj

        shapes = getattr(var_cfg_obj, "shapes", None)
        if not isinstance(shapes, list):
            return var_cfg_obj

        new_shapes: List[MembershipShape] = []
        for s in shapes:
            pts = ov.get(s.label)
            if isinstance(pts, list) and all(isinstance(x, (int, float)) for x in pts):
                new_pts = tuple(float(x) for x in pts)
            else:
                new_pts = s.points
            new_shapes.append(MembershipShape(s.label, s.shape, new_pts))
        return VariableConfig(getattr(var_cfg_obj, "name", var_key), new_shapes)

    for k in ["water_temperature", "pH", "flow_ratio", "turbidity", "air_temperature", "humidity"]:
        if k in variables:
            variables[k] = _rebuilt_var(k, variables[k])

    # EC has mode split
    ec = variables.get("EC")
    if isinstance(ec, dict):
        if "water" in ec:
            ec["water"] = _rebuilt_var("EC_water", ec["water"])
        if "hydro" in ec:
            ec["hydro"] = _rebuilt_var("EC_hydro", ec["hydro"])


def _apply_rule_weight_overrides(cfg: Dict[str, object], overrides: Dict[str, object]) -> None:
    rw = overrides.get("rule_weights")
    if not isinstance(rw, dict):
        return
    cur = cfg.get("rule_weights")
    if not isinstance(cur, dict):
        return
    for k, v in rw.items():
        if k in cur and isinstance(v, (int, float)):
            cur[k] = float(v)


def load_config(state_path: str | None = None, crop_config_path: str | None = None) -> Dict[str, object]:
    """
    Load baseline config, then apply crop profile overrides, then improvement overrides.

    Layer order: base → crop_config.json → improvement_state.json
    """
    base = default_config()
    imp = base.get("improvement") if isinstance(base.get("improvement"), dict) else {}
    sp = state_path or str((imp or {}).get("state_path", DEFAULT_STATE_PATH))
    cp = crop_config_path or CROP_CONFIG_PATH

    cfg = copy.deepcopy(base)

    # Layer 1: Crop profile overrides
    crop_p = Path(cp)
    if crop_p.exists():
        try:
            crop_data = json.loads(crop_p.read_text(encoding="utf-8"))
            if isinstance(crop_data, dict):
                ov = crop_data.get("overrides", {}) or {}
                _apply_membership_overrides(cfg, ov)
                _apply_rule_weight_overrides(cfg, ov)
                if "ec_mode" in crop_data:
                    cfg["ec_mode_active"] = crop_data["ec_mode"]
                if "control_targets" in crop_data:
                    cfg["control_targets"] = crop_data["control_targets"]
                if "improvement_limits" in crop_data:
                    imp_cfg = cfg.get("improvement") or {}
                    imp_cfg["membership_delta_cap"] = crop_data["improvement_limits"]
                    cfg["improvement"] = imp_cfg
                cfg["_crop_loaded"] = crop_data.get("crop", "unknown")
        except Exception:
            pass

    # Layer 2: UCB1 bandit improvement overrides
    p = Path(sp)
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                ov = data.get("overrides", {}) or {}
                _apply_membership_overrides(cfg, ov)
                _apply_rule_weight_overrides(cfg, ov)
        except Exception:
            pass

    return cfg
