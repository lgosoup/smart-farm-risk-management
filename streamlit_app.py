from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st

import matplotlib as mpl
from matplotlib import font_manager

BASE_DIR = Path(__file__).resolve().parent / "fuzzyLozic"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import load_config  # noqa: E402
from fuzzify import (
    compute_delta_flags,
    compute_integrity_flags,
    compute_timeseries_flags,
    fuzzify_inputs,
    preprocess_inputs,
)  # noqa: E402
from inference import infer  # noqa: E402
from membership import evaluate  # noqa: E402
from rules import build_rules  # noqa: E402


HISTORY_INPUTS_PATH = Path(__file__).resolve().parent / "data" / "history_inputs.jsonl"
DEFAULT_INPUTS_PATH = Path(__file__).resolve().parent / "data" / "inputs.json"

VAR_INFO = {
    "water_temperature": {"label": "수온", "unit": "°C", "cmap": "YlOrRd"},
    "pH": {"label": "pH", "unit": "", "cmap": "Greens"},
    "EC": {"label": "EC", "unit": "mS/cm", "cmap": "Purples"},
    "flow_ratio": {"label": "유량비", "unit": "", "cmap": "Oranges"},
    "turbidity": {"label": "탁도", "unit": "NTU", "cmap": "Greys"},
    "air_temperature": {"label": "공기 온도", "unit": "°C", "cmap": "RdPu"},
    "humidity": {"label": "습도", "unit": "%", "cmap": "Blues"},
}


def _load_default_inputs() -> Dict[str, float]:
    if DEFAULT_INPUTS_PATH.exists():
        data = json.loads(DEFAULT_INPUTS_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}
    return {
        "water_temperature": 18.0, "pH": 6.3, "EC": 0.5,
        "flow_ratio": 1.0, "turbidity": 1.0,
        "air_temperature": 25.0, "humidity": 60.0,
    }


def _read_history_inputs(path: Path) -> Tuple[List[Dict[str, Optional[float]]], List[str]]:
    if not path.exists():
        return [], []
    rows: List[Dict[str, Optional[float]]] = []
    ts_labels: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            sensor = obj.get("sensor")
            if isinstance(sensor, dict):
                rows.append(
                    {k: (float(v) if isinstance(v, (int, float)) else None) for k, v in sensor.items()}
                )
                ts = obj.get("ts")
                ts_labels.append(str(ts) if isinstance(ts, str) else "")
    return rows, ts_labels


def _get_var_cfg(cfg: Dict[str, object], var_name: str, ec_mode: str):
    variables = cfg.get("variables", {})
    if var_name == "EC":
        return variables.get("EC", {}).get(ec_mode)
    return variables.get(var_name)


def _infer_ranges(cfg: Dict[str, object]) -> Dict[str, Tuple[float, float]]:
    ranges: Dict[str, Tuple[float, float]] = {}
    integrity = cfg.get("integrity_policy", {}) or {}
    integrity_ranges = integrity.get("ranges", {}) or {}
    for var, bounds in integrity_ranges.items():
        if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
            ranges[var] = (float(bounds[0]), float(bounds[1]))
    return ranges


def _shape_range(var_cfg) -> Tuple[float, float]:
    if not var_cfg:
        return 0.0, 1.0
    all_pts: List[float] = []
    for shape in var_cfg.shapes:
        all_pts.extend(list(shape.points))
    if not all_pts:
        return 0.0, 1.0
    return float(min(all_pts)), float(max(all_pts))


def _ideal_range(var_cfg) -> Optional[Tuple[float, float]]:
    if not var_cfg:
        return None
    candidates: List[Tuple[float, float]] = []
    for shape in var_cfg.shapes:
        if _is_ideal_label(shape.label):
            candidates.append((float(min(shape.points)), float(max(shape.points))))
    if not candidates:
        return None
    lo = min(a for a, _ in candidates)
    hi = max(b for _, b in candidates)
    return lo, hi


def _is_ideal_label(label: str) -> bool:
    text = str(label)
    keywords = ["적정", "안정", "최적", "목표", "맑음"]
    return any(k in text for k in keywords)


def _fmt_range(bounds: Optional[Tuple[float, float]], digits: int = 2) -> str:
    if not bounds:
        return "-"
    lo, hi = bounds
    fmt = f"{{:.{digits}f}}"
    return f"{fmt.format(lo)} ~ {fmt.format(hi)}"


def _var_label(var: str) -> str:
    return VAR_INFO.get(var, {}).get("label", var)


def _var_unit(var: str) -> str:
    return VAR_INFO.get(var, {}).get("unit", "")


def _range_hint(var_key: str, var_cfg, ranges: Dict[str, Tuple[float, float]]) -> str:
    ideal = _ideal_range(var_cfg)
    unit = _var_unit(var_key)
    unit_text = f" ({unit})" if unit else ""
    if ideal:
        return f"적정 범위{unit_text}: {_fmt_range(ideal)}"
    return f"적정 범위{unit_text}: -"


def _plot_memberships(var_cfg, current_value: float, var_key: str):
    import matplotlib.pyplot as plt

    _set_korean_font()

    xs_min, xs_max = _shape_range(var_cfg)
    pad = (xs_max - xs_min) * 0.1 if xs_max > xs_min else 1.0
    start = xs_min - pad
    end = xs_max + pad
    step = (end - start) / 200 if end > start else 0.1

    xs: List[float] = []
    x = start
    while x <= end + 1e-9:
        xs.append(float(x))
        x += step

    fig, ax = plt.subplots(figsize=(6.2, 3.6))

    ideal = _ideal_range(var_cfg)
    if ideal:
        ax.axvspan(ideal[0], ideal[1], color="#ffd166", alpha=0.15, label="적정 구간")

    mu_now: List[Tuple[str, float]] = []
    for shape in var_cfg.shapes:
        mu_now.append((shape.label, evaluate(shape.shape, shape.points, current_value)))
    top_labels = {lbl for lbl, _ in sorted(mu_now, key=lambda x: x[1], reverse=True)[:2]}

    for idx, shape in enumerate(var_cfg.shapes):
        ys = [evaluate(shape.shape, shape.points, xv) for xv in xs]
        color = _label_color(shape.label)
        lw = 2.6 if shape.label in top_labels else 1.2
        alpha = 0.9 if shape.label in top_labels else 0.6
        ax.plot(xs, ys, label=_display_label(shape.label), color=color, linewidth=lw, alpha=alpha)
        if shape.label in top_labels:
            ax.fill_between(xs, 0, ys, color=color, alpha=0.08)

    ax.axvline(current_value, color="black", linestyle="--", linewidth=1)
    for lbl, mu in mu_now:
        if lbl in top_labels and mu > 0:
            ax.scatter([current_value], [mu], color="black", s=30, zorder=5)
            ax.text(
                current_value,
                mu + 0.05,
                f"{_display_label(lbl)} ({mu:.2f})",
                fontsize=9,
                ha="center",
            )
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(var_cfg.name)
    ax.set_ylabel("소속도(0~1)")
    ax.legend(fontsize="small", ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=True)
    fig.subplots_adjust(bottom=0.28)
    st.pyplot(fig, clear_figure=True)


def _set_korean_font() -> None:
    preferred = ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "DejaVu Sans"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in available:
            mpl.rcParams["font.family"] = name
            mpl.rcParams["axes.unicode_minus"] = False
            return
    mpl.rcParams["axes.unicode_minus"] = False


def _format_flags(flags: Dict[str, object]) -> List[str]:
    if not flags:
        return []
    return [str(k) for k, v in flags.items() if v]


def _format_spike_flags(spike: Dict[str, Dict[str, object]]) -> List[str]:
    rows: List[str] = []
    for var, info in spike.items():
        direction = info.get("direction")
        delta = info.get("delta")
        if direction:
            rows.append(f"{var}: {direction} ({delta:+.3f})")
        else:
            rows.append(f"{var}")
    return rows


def _score_membership_table(score_membership: Dict[str, float]) -> Dict[str, float]:
    return {k: float(v) for k, v in sorted(score_membership.items(), key=lambda kv: kv[1], reverse=True)}


def _display_label(label: str) -> str:
    label_map = {
        # risk/output labels
        "留ㅼ슦?덉젙": "매우 안전",
        "?덉젙": "안전",
        "寃쎄퀎": "경계",
        "二쇱쓽": "주의",
        "寃쎄퀬": "경고",
        "?꾪뿕": "위험",
        # temperature
        "留ㅼ슦??쓬": "매우 낮음",
        "??쓬": "낮음",
        "?덉젙(理쒖쟻)": "적정(최적)",
        "寃쎄퀎(?곸듅)": "경계(고온)",
        "寃쎄퀬(怨좎삩)": "경고(고온)",
        "?꾪뿕(湲됯퀬??": "위험(극고온)",
        # DO
        "移섎챸????쓬": "치명적 저산소",
        "?덉젙(紐⑺몴)": "적정(목표)",
        "異⑸텇(?믪쓬)": "충분(높음)",
        # pH
        "媛뺤궛???꾪뿕)": "강산성(위험)",
        "?쎌궛??寃쎄퀬?볦＜??": "약산성(경계)",
        "?쎌븣移쇰━(寃쎄퀬?볦＜??": "약알칼리(경계)",
        "媛뺤븣移쇰━(?꾪뿕)": "강알칼리(위험)",
        # FR
        "?뺤껜(?꾪뿕)": "정체(위험)",
        "留ㅼ슦??쓬(寃쎄퀬)": "매우 낮음(경고)",
        "??쓬(二쇱쓽)": "낮음(주의)",
        "怨쇰떎(二쇱쓽)": "과다(주의)",
        # EC
        "?믪쓬(寃쎄퀬)": "높음(경고)",
        "留ㅼ슦?믪쓬(?꾪뿕)": "매우 높음(위험)",
        # Turbidity
        "留묒쓬(?덉젙)": "맑음(적정)",
        "?쎄컙 ?곹븿(寃쎄퀎)": "약간 상승(경계)",
        "?곹븿(二쇱쓽)": "상승(주의)",
        "留ㅼ슦 ?곹븿(寃쎄퀬)": "매우 상승(경고)",
        "洹밸룄濡??곹븿(?꾪뿕)": "극심 상승(위험)",
    }
    return label_map.get(label, label)


def _display_map_dict(values: Dict[str, float]) -> Dict[str, float]:
    return {_display_label(k): float(v) for k, v in values.items()}


def _memberships_table(memberships: Dict[str, float]) -> List[Dict[str, object]]:
    rows = [
        {"label": _display_label(k), "degree": float(v)}
        for k, v in sorted(memberships.items(), key=lambda kv: kv[1], reverse=True)
    ]
    return rows


def _label_color(label: str) -> str:
    disp = _display_label(label)
    # Safe/ideal
    if any(k in disp for k in ["안정", "적정", "최적", "목표", "맑음"]):
        return "#2E7D32"  # green
    # Boundary / caution
    if any(k in disp for k in ["경계", "주의"]):
        return "#F9A825"  # amber
    # Warning / danger
    if "경고" in disp:
        return "#EF6C00"  # orange
    if any(k in disp for k in ["위험", "치명", "극"]):
        return "#C62828"  # red
    # Low/High extremes (tend to be riskier than neutral)
    if any(k in disp for k in ["매우 낮음", "낮음", "과다", "높음"]):
        return "#D32F2F"  # red-ish
    return "#546E7A"  # fallback gray-blue


def _rule_name(rule_id: str) -> str:
    mapping = {
        "R2": "고수온 극고",
        "R3": "유량 정체",
        "R4": "pH 극단",
        "R5": "탁도 극심",
        "R6": "고수온+저유량",
        "R7": "고EC+저유량",
        "R8": "저유량+탁도상승",
        "R9": "고EC+유량경계",
        "R10": "수온/유량 경계",
        "R11": "복합 경계",
        "R12": "경고 누적 승격",
        "R13": "전반 최적",
        "R14": "기본 최적",
        "R16": "고수온 경고",
        "R17": "저유량 경고",
        "R18": "탁도 경고",
        "R19": "고EC 경고",
        "R20": "pH 경계",
        "DELTA": "급변 플래그",
        "TS_TREND": "추세 플래그",
        "TS_VOL": "변동성 플래그",
        "TS_PERSIST": "지속 플래그",
        "TS_SPIKE": "스파이크 플래그",
        "TS_RECOVERY_FAIL": "회복실패 플래그",
    }
    return mapping.get(rule_id, rule_id)


def _describe_ts_flag(flag: str, cfg: Dict[str, object]) -> Dict[str, str]:
    ts_cfg = cfg.get("timeseries_policy", {}) or {}
    window = int(ts_cfg.get("window", 0))

    def _detail(text: str) -> str:
        return f"{text} (window={window})" if window else text

    parts = flag.split("_")
    if len(parts) < 3:
        return {"flag": flag, "variable": "-", "meaning": "알 수 없음", "detail": "-"}

    kind = parts[1]
    var = parts[2]
    var_label = _var_label(var)

    if kind == "TREND":
        direction = parts[3] if len(parts) > 3 else ""
        fast = "FAST" in parts
        spec = (ts_cfg.get("trend", {}) or {}).get(var, {})
        th = spec.get("threshold")
        fast_th = spec.get("fast_threshold")
        dir_text = "상승" if direction == "UP" else "하강"
        fast_text = " (빠름)" if fast else ""
        detail = f"변화량 ≥ {th}" if th is not None else "변화량 기준"
        if fast and fast_th is not None:
            detail += f", 빠름 ≥ {fast_th}"
        return {
            "flag": flag,
            "variable": var_label,
            "meaning": f"추세 {dir_text}{fast_text}",
            "detail": _detail(detail),
        }

    if kind == "VOL":
        th = (ts_cfg.get("volatility", {}) or {}).get(var)
        detail = f"표준편차 ≥ {th}" if th is not None else "표준편차 기준"
        return {
            "flag": flag,
            "variable": var_label,
            "meaning": "변동성 높음",
            "detail": _detail(detail),
        }

    if kind == "PERSIST":
        level = parts[3] if len(parts) > 3 else ""
        spec = (ts_cfg.get("persistence", {}) or {}).get(var, {})
        th = spec.get("threshold")
        cnt = spec.get("count")
        op = spec.get("op")
        lvl = "높음" if level == "HIGH" else "낮음"
        detail = f"{cnt}회 연속 {op} {th}" if th is not None else "지속 기준"
        return {
            "flag": flag,
            "variable": var_label,
            "meaning": f"지속 {lvl}",
            "detail": _detail(detail),
        }

    if kind == "SPIKE":
        direction = parts[3] if len(parts) > 3 else ""
        spec = (ts_cfg.get("spike", {}) or {}).get(var, {})
        th = spec.get("up" if direction == "UP" else "down")
        dir_text = "급등" if direction == "UP" else "급락"
        detail = f"직전 대비 Δ ≥ {th}" if th is not None else "급변 기준"
        return {
            "flag": flag,
            "variable": var_label,
            "meaning": f"스파이크 {dir_text}",
            "detail": _detail(detail),
        }

    if kind == "RECOVERY" and len(parts) > 3 and parts[2] == "FAIL":
        var = parts[3] if len(parts) > 3 else var
        var_label = _var_label(var)
        spec = (ts_cfg.get("recovery_fail", {}) or {}).get(var, {})
        th = spec.get("bad_threshold")
        cnt = spec.get("count")
        min_rec = spec.get("min_recover_delta")
        detail = f"{cnt}회 연속 bad(≤/≥ {th}), 회복폭 < {min_rec}" if th is not None else "회복 실패 기준"
        return {
            "flag": flag,
            "variable": var_label,
            "meaning": "회복 실패",
            "detail": _detail(detail),
        }

    return {"flag": flag, "variable": var_label, "meaning": "시계열 플래그", "detail": _detail("-")}


def _build_flag_rows(
    cfg: Dict[str, object],
    delta_flags: Dict[str, bool],
    ts_flags: Dict[str, bool],
    integrity_flags: Dict[str, object],
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []

    delta_cfg = cfg.get("delta_policy", {}) or {}
    delta_th = delta_cfg.get("thresholds", {}) or {}
    for var, on in delta_flags.items():
        if not on:
            continue
        th = delta_th.get(var)
        detail = f"Δ ≥ {th}" if th is not None else "급변 기준"
        rows.append(
            {
                "type": "Delta",
                "flag": var,
                "variable": _var_label(var),
                "meaning": "전 단계 대비 급변",
                "detail": detail,
            }
        )

    for flag in ts_flags.keys():
        desc = _describe_ts_flag(flag, cfg)
        rows.append(
            {
                "type": "시계열",
                "flag": desc["flag"],
                "variable": desc["variable"],
                "meaning": desc["meaning"],
                "detail": desc["detail"],
            }
        )

    out_of_range = integrity_flags.get("out_of_range", {}) if isinstance(integrity_flags, dict) else {}
    for var, on in out_of_range.items():
        if not on:
            continue
        ranges = (cfg.get("integrity_policy", {}) or {}).get("ranges", {}) or {}
        bounds = ranges.get(var)
        detail = f"허용 {bounds[0]}~{bounds[1]}" if isinstance(bounds, (list, tuple)) else "허용 범위 초과"
        rows.append(
            {
                "type": "무결성",
                "flag": f"OUT_{var}",
                "variable": _var_label(var),
                "meaning": "허용 범위 초과",
                "detail": detail,
            }
        )

    spike = integrity_flags.get("spike", {}) if isinstance(integrity_flags, dict) else {}
    spike_th = (cfg.get("integrity_policy", {}) or {}).get("spike_thresholds", {}) or {}
    for var, info in spike.items():
        if not isinstance(info, dict):
            continue
        direction = info.get("direction")
        delta = info.get("delta")
        th = spike_th.get(var)
        detail = f"Δ {delta:+.2f}, 기준 {th}" if th is not None else f"Δ {delta:+.2f}"
        rows.append(
            {
                "type": "무결성",
                "flag": f"SPIKE_{var}",
                "variable": _var_label(var),
                "meaning": f"급변({direction})" if direction else "급변",
                "detail": detail,
            }
        )

    return rows


def _fired_rules_table(fired_rules: List[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for r in sorted(fired_rules, key=lambda x: float(x.get("firing_strength", 0.0)), reverse=True):
        rows.append(
            {
                "rule": _rule_name(str(r.get("rule_id"))),
                "strength": round(float(r.get("firing_strength", 0.0)), 4),
                "label": r.get("conclusion_label"),
                "reason": r.get("human_readable_reason"),
            }
        )
    return rows


def main() -> None:
    st.set_page_config(page_title="Smart Farm Fuzzy Risk Demo", layout="wide")
    st.title("Smart Farm Risk Management System – Fuzzy Logic Demo")
    st.caption("입력값을 즉시 수정하고, 퍼지로직의 중간 과정과 위험도 결과를 확인할 수 있습니다.")

    cfg = load_config()  # data/improvement_state.json 사용 (config.py 기본값)
    default_inputs = _load_default_inputs()
    ranges = _infer_ranges(cfg)

    st.sidebar.header("입력값")
    _ec_options = ["water", "hydro"]
    _active_ec = cfg.get("ec_mode_active", "water")
    _ec_default_idx = _ec_options.index(_active_ec) if _active_ec in _ec_options else 0
    ec_mode = st.sidebar.selectbox("EC 모드", _ec_options, index=_ec_default_idx)

    def _num_input(key: str, label: str, default: float, step: float, var_cfg) -> float:
        lo, hi = ranges.get(key, (-1000.0, 1000.0))
        span = hi - lo
        if span > 0:
            lo -= span * 0.2
            hi += span * 0.2
        value = float(
            st.sidebar.number_input(
                label,
                min_value=float(lo),
                max_value=float(hi),
                value=float(default),
                step=float(step),
                format="%.3f",
                key=f"input_{key}",
            )
        )
        st.sidebar.caption(_range_hint(key, var_cfg, ranges))
        return value

    inputs = {
        "water_temperature": _num_input(
            "water_temperature", "수온 (°C)",
            default_inputs.get("water_temperature", 18.0), 0.1,
            _get_var_cfg(cfg, "water_temperature", ec_mode),
        ),
        "pH": _num_input("pH", "pH", default_inputs.get("pH", 6.3), 0.01, _get_var_cfg(cfg, "pH", ec_mode)),
        "EC": _num_input("EC", "EC (mS/cm)", default_inputs.get("EC", 0.5), 0.01, _get_var_cfg(cfg, "EC", ec_mode)),
        "flow_ratio": _num_input(
            "flow_ratio", "유량비",
            default_inputs.get("flow_ratio", 1.0), 0.01,
            _get_var_cfg(cfg, "flow_ratio", ec_mode),
        ),
        "turbidity": _num_input(
            "turbidity", "탁도 (NTU)",
            default_inputs.get("turbidity", 1.0), 0.1,
            _get_var_cfg(cfg, "turbidity", ec_mode),
        ),
        "air_temperature": _num_input(
            "air_temperature", "공기 온도 (°C)",
            default_inputs.get("air_temperature", 25.0), 0.1,
            _get_var_cfg(cfg, "air_temperature", ec_mode),
        ),
        "humidity": _num_input(
            "humidity", "습도 (%)",
            default_inputs.get("humidity", 60.0), 0.5,
            _get_var_cfg(cfg, "humidity", ec_mode),
        ),
    }

    st.sidebar.header("히스토리/시계열")
    history_source = st.sidebar.selectbox("히스토리 소스", ["파일(history_inputs.jsonl)", "세션(임시)"], index=0)
    auto_append = st.sidebar.checkbox("현재 입력을 세션 히스토리에 추가", value=True)
    clear_session = st.sidebar.button("세션 히스토리 초기화")

    if "session_history" not in st.session_state:
        st.session_state.session_history = []

    if clear_session:
        st.session_state.session_history = []

    history_rows, history_ts = _read_history_inputs(HISTORY_INPUTS_PATH)
    if history_source.startswith("세션"):
        history_inputs = st.session_state.session_history[:]
        if auto_append:
            history_inputs = history_inputs + [inputs]
    else:
        history_inputs = history_rows

    processed = preprocess_inputs(inputs, cfg, history=history_inputs)
    memberships = fuzzify_inputs(processed, cfg, ec_mode=ec_mode)

    previous_inputs = history_inputs[-1] if history_inputs else None
    delta_flags = compute_delta_flags(processed, previous_inputs, cfg)
    integrity_flags = compute_integrity_flags(processed, previous_inputs, cfg)
    ts_flags = compute_timeseries_flags(processed, history_inputs, cfg)

    merged_flags: Dict[str, bool] = {}
    merged_flags.update({k: bool(v) for k, v in (delta_flags or {}).items()})
    merged_flags.update({k: True for k in ts_flags.keys()})

    risk_score, risk_label, score_membership, label_peaks, fired_rules = infer(
        memberships, cfg, delta_flags=merged_flags
    )

    if history_source.startswith("세션") and auto_append:
        st.session_state.session_history = history_inputs

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("현재 입력값")
        rows = []
        for var in ["water_temperature", "pH", "EC", "flow_ratio", "turbidity", "air_temperature", "humidity"]:
            var_cfg = _get_var_cfg(cfg, var, ec_mode)
            rows.append(
                {
                    "항목": f"{_var_label(var)} ({var})",
                    "현재값": processed.get(var),
                    "단위": _var_unit(var),
                    "적정 범위": _fmt_range(_ideal_range(var_cfg)),
                }
            )
        st.dataframe(rows, use_container_width=True)
        with st.expander("원본 입력 JSON"):
            st.json(processed)

    with col2:
        st.subheader("위험도 결과")
        st.metric("Risk Score", f"{risk_score:.2f}")
        st.write(f"**Risk Label:** {_display_label(risk_label)}")
        st.write("라벨별 점수 membership")
        st.bar_chart(_display_map_dict(_score_membership_table(score_membership)))
        st.write("라벨별 출력 피크(규칙 출력 최대값)")
        st.bar_chart(_display_map_dict(_score_membership_table(label_peaks)))

    st.subheader("퍼지 멤버십 결과")
    _FUZZY_VARS = ["water_temperature", "pH", "EC", "flow_ratio", "turbidity", "air_temperature", "humidity"]
    _TAB_NAMES = ["수온", "pH", "EC", "유량비", "탁도", "공기온도", "습도"]
    tabs = st.tabs(_TAB_NAMES)
    for tab, var in zip(tabs, _FUZZY_VARS):
        with tab:
            var_cfg = _get_var_cfg(cfg, var, ec_mode)
            if not var_cfg:
                st.info("멤버십 설정이 없습니다.")
                continue
            st.markdown(
                f"**{_var_label(var)} ({var})** | 단위: {_var_unit(var) or '-'} | "
                f"적정 {_fmt_range(_ideal_range(var_cfg))}"
            )
            st.caption("그래프 해석: x축=값, y축=소속도(0~1), 점선=현재 입력, 음영=적정 구간, 굵은 선=현재 입력에서 높은 멤버십")
            _plot_memberships(var_cfg, float(inputs[var]), var)
            st.write("현재 값의 멤버십 분포")
            st.bar_chart(_display_map_dict(memberships.get(var, {})))
            st.write("현재 값의 멤버십 정도(내림차순)")
            st.dataframe(_memberships_table(memberships.get(var, {})), use_container_width=True)

    st.subheader("규칙 발화(Fired Rules)")
    if fired_rules:
        st.dataframe(_fired_rules_table(fired_rules), use_container_width=True)
    else:
        st.info("발화된 규칙이 없습니다.")

    st.subheader("시계열/무결성 플래그")
    flag_rows = _build_flag_rows(cfg, delta_flags, ts_flags, integrity_flags)
    if flag_rows:
        st.dataframe(flag_rows, use_container_width=True)
    else:
        st.info("현재 활성 플래그가 없습니다.")

    with st.expander("플래그 해설"):
        st.markdown(
            "- **Delta**: 전 단계 대비 급격 변화\n"
            "- **시계열**: 최근 window 구간의 추세/변동성/지속/스파이크/회복 실패 감지\n"
            "- **무결성**: 입력 허용 범위 초과 또는 급격한 스파이크"
        )

    st.subheader("히스토리 입력 샘플")
    if history_inputs:
        show_n = st.slider("표시할 최근 샘플 수", min_value=3, max_value=30, value=10)
        st.dataframe(history_inputs[-show_n:], use_container_width=True)
    else:
        st.info("히스토리 입력이 없습니다.")

    with st.expander("룰 베이스 보기"):
        rules = build_rules(cfg)
        rows = [
            {
                "rule_id": r.rule_id,
                "conclusion": r.conclusion_label,
                "reason": r.human_readable_reason,
            }
            for r in rules.values()
        ]
        st.dataframe(rows, use_container_width=True)


if __name__ == "__main__":
    main()
