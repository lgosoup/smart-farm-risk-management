"""넷제로 스마트팜 데모 페이지.

실행: streamlit run netzero_app.py

계산 함수(run_comparison 등)는 streamlit 없이도 import해서 쓸 수 있다.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from netzero.controllers import CONTROLLER_DESCRIPTIONS, DEFAULT_CONTROLLERS
from netzero.experiment import CONTROLLER_LABELS, EpisodeResult, Event, run_episode, summary_rows
from netzero.facility import available_facilities, load_facility
from netzero.fuzzy import available_crops
from netzero.grid import load_grid
from netzero.simulator import CLIMATE_DIR, available_climates

GRID_NAME = "kr_stylized"

FUNCTION_LABELS = {
    "water_temp_down": "수온 하강",
    "water_temp_up": "수온 상승",
    "flow_up": "유량 증가",
    "ec_up": "EC 증가",
    "ec_down": "EC 감소",
    "turbidity_down": "탁도 감소",
    "oxygen_supply": "산소 공급",
    "air_exchange": "환기",
}

VAR_LABELS = {
    "water_temperature": "수온",
    "flow_ratio": "유량비",
    "EC": "EC",
    "turbidity": "탁도",
    "pH": "pH",
}

SUMMARY_COLUMNS = {
    "facility": "시설",
    "controller": "운전 방식",
    "in_range_%": "적정 범위 유지율(%)",
    "alert_%": "경고 비율(%)",
    "danger_h": "위험 시간(h)",
    "energy_kWh": "에너지(kWh)",
    "energy_saving_%": "에너지 절감률(%)",
    "carbon_kg": "탄소(kgCO2e)",
    "carbon_reduction_%": "탄소 감축률(%)",
    "kgCO2/kWh": "평균 배출계수(kg/kWh)",
    "cost_KRW": "전기요금(원)",
    "pred_mae_first→last": "수온 예측오차(첫날→마지막날)",
    "adapt_h": "적응 시간(h)",
}


# ----------------------------------------------------------------------
# 계산 (streamlit 불필요)
# ----------------------------------------------------------------------
def build_events(
    degrade: Optional[Tuple[str, float, int]] = None,
    remove: Optional[Tuple[str, int]] = None,
) -> List[Event]:
    events: List[Event] = []
    if degrade:
        dev, factor, day = degrade
        events.append(Event(day=int(day), type="degrade", device=str(dev), factor=float(factor)))
    if remove:
        dev, day = remove
        events.append(Event(day=int(day), type="remove", device=str(dev)))
    return events


def run_comparison(
    facility: str,
    controllers: Sequence[str],
    crop: str,
    climate: str,
    days: int,
    seed: int,
    events: Sequence[Event] = (),
    grid: str = GRID_NAME,
) -> List[Dict[str, object]]:
    """제어기별 에피소드를 실행하고 dict 리스트로 돌려준다 (캐시 가능한 형태)."""
    out = []
    for c in controllers:
        r = run_episode(facility, c, crop=crop, climate=climate, grid=grid, days=days, seed=seed, events=list(events))
        out.append(r.to_dict(with_series=True))
    return out


def results_from_dicts(dicts: Sequence[Dict[str, object]]) -> List[EpisodeResult]:
    res = []
    for d in dicts:
        res.append(EpisodeResult(
            facility=d["facility"], controller=d["controller"], controller_name=d["controller_name"],
            crop=d["crop"], climate=d["climate"], days=d["days"], seed=d["seed"], metrics=d["metrics"],
            series=d.get("series") or {}, learned_coefficients=d.get("learned_coefficients"),
            runtime_sec=float(d.get("runtime_sec") or 0.0),
        ))
    return res


def device_table(facility_name: str) -> pd.DataFrame:
    fac = load_facility(facility_name)
    rows = []
    for d in fac.devices:
        if d.unit == "min":
            levels = ", ".join(f"{x:g}" for x in d.levels) + " 분/시간"
            power = f"{d.power_w:g} W"
        else:
            levels = ", ".join(f"{x:g}" for x in d.levels) + f" {d.unit}/시간"
            power = f"{d.energy_wh_per_unit:g} Wh/{d.unit}"
        rows.append({
            "ID": d.id,
            "이름": d.name,
            "기능": ", ".join(FUNCTION_LABELS.get(f, f) for f in d.functions),
            "제어 단계": levels,
            "소비전력": power,
        })
    return pd.DataFrame(rows)


def _nm(code: str) -> str:
    return CONTROLLER_LABELS.get(code, code)


def summary_table(results: Sequence[EpisodeResult]) -> pd.DataFrame:
    df = pd.DataFrame(summary_rows(results))
    if df.empty:
        return df
    df = df.rename(columns=SUMMARY_COLUMNS)
    if "시설" in df.columns:
        df = df.drop(columns=["시설"])
    return df


def d_vs_a(results: Sequence[EpisodeResult]) -> Optional[Dict[str, float]]:
    by = {r.controller: r for r in results}
    a, d = by.get("A"), by.get("D")
    if a is None or d is None:
        return None
    ma, md = a.metrics, d.metrics
    out: Dict[str, float] = {}
    out["energy_saving_%"] = 100.0 * (ma["energy_kwh"] - md["energy_kwh"]) / ma["energy_kwh"] if ma["energy_kwh"] else 0.0
    out["carbon_reduction_%"] = 100.0 * (ma["carbon_kg"] - md["carbon_kg"]) / ma["carbon_kg"] if ma["carbon_kg"] else 0.0
    out["in_range_D_%"] = 100.0 * md["in_range_rate"]
    out["in_range_diff_pp"] = 100.0 * (md["in_range_rate"] - ma["in_range_rate"])
    return out


def water_temp_frame(results: Sequence[EpisodeResult]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    s0 = results[0].series
    df = pd.DataFrame({"외기 온도": s0.get("outdoor", [])}, index=s0.get("hour", []))
    for r in results:
        df[f"{_nm(r.controller)} 수온"] = r.series.get("water_temperature", [])
    df.index.name = "시간(h)"
    return df


def cumulative_carbon_frame(results: Sequence[EpisodeResult]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(index=results[0].series.get("hour", []))
    for r in results:
        df[_nm(r.controller)] = pd.Series(r.series.get("carbon_kg", []), index=df.index).cumsum()
    df.index.name = "시간(h)"
    return df


def grid_ef_frame(grid: str = GRID_NAME) -> pd.DataFrame:
    g = load_grid(grid)
    df = pd.DataFrame({"배출계수(kg/kWh)": g.emission_factor}, index=list(range(24)))
    df.index.name = "시각"
    return df


def pred_error_frame(results: Sequence[EpisodeResult], window: int = 24) -> pd.DataFrame:
    """학습형 제어기(C, D)의 수온 예측오차 24시간 이동평균."""
    cols = {}
    for r in results:
        if r.controller not in ("C", "D"):
            continue
        err = r.series.get("pred_error_wt") or []
        if not err:
            continue
        cols[_nm(r.controller)] = pd.Series(err).rolling(window, min_periods=1).mean()
    if not cols:
        return pd.DataFrame()
    df = pd.DataFrame(cols)
    df.index.name = "시간(h)"
    return df


def coefficient_table(results: Sequence[EpisodeResult], facility_name: str) -> pd.DataFrame:
    """학습 계수와 프로필 사전값 비교."""
    fac = load_facility(facility_name)
    priors: Dict[Tuple[str, str], float] = {}
    dev_names: Dict[str, str] = {}
    for d in fac.devices:
        for e in d.effects:
            nm = d.id if e.model != "exchange" else f"{d.id}:exchange"
            priors[(e.var, nm)] = e.prior
            dev_names[nm] = d.name + (" (외기 교환)" if e.model == "exchange" else "")
    priors[("water_temperature", "ambient")] = fac.prior_thermal_coupling
    term_names = {"bias": "상수 변화", "ambient": "외기 열전달", "sun": "일사"}

    learned = {r.controller: r.learned_coefficients for r in results if r.controller in ("C", "D") and r.learned_coefficients}
    if not learned:
        return pd.DataFrame()
    keys: List[Tuple[str, str]] = []
    for coefs in learned.values():
        for var, terms in coefs.items():
            for nm in terms:
                if (var, nm) not in keys:
                    keys.append((var, nm))
    rows = []
    for var, nm in keys:
        prior = priors.get((var, nm), 0.0 if nm in term_names else None)
        row: Dict[str, object] = {
            "변수": VAR_LABELS.get(var, var),
            "항목": dev_names.get(nm, term_names.get(nm, nm)),
            "사전값(프로필)": prior,
        }
        for c, coefs in learned.items():
            v = (coefs.get(var) or {}).get(nm)
            row[f"{_nm(c)} 학습값"] = v
            if v is not None and prior not in (None, 0.0):
                row[f"{_nm(c)} 학습/사전"] = round(v / prior, 2)
        rows.append(row)
    return pd.DataFrame(rows)


def last_commands(results: Sequence[EpisodeResult], facility_name: str, events: Sequence[Event] = ()) -> Tuple[Optional[str], Dict[str, object]]:
    """D(없으면 마지막 선택 제어기)의 마지막 스텝 제어값을 장치 명령으로 변환."""
    if not results:
        return None, {}
    by = {r.controller: r for r in results}
    r = by.get("D") or results[-1]
    controls = r.series.get("controls") or []
    if not controls:
        return r.controller, {}
    fac = load_facility(facility_name)
    for ev in events:
        if ev.type == "remove" and ev.day * 24 < len(controls):
            try:
                fac = fac.without(ev.device)
            except Exception:
                pass
    return r.controller, fac.build_commands(controls[-1])


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="넷제로 스마트팜", layout="wide")

    @st.cache_data(show_spinner=False)
    def cached_run(facility, controllers, crop, climate, days, seed, degrade, remove):
        events = build_events(degrade, remove)
        return run_comparison(facility, list(controllers), crop, climate, days, seed, events)

    st.title("넷제로 스마트팜: 탄소 인지형 제어 비교")
    st.caption("같은 가상 농장에서 제어 방식 A~D를 돌려 에너지·탄소·작물 환경을 비교한다.")

    facilities = available_facilities()
    fac_names = {f: load_facility(f).name for f in facilities}
    climates = available_climates()
    clim_names = {}
    for c in climates:
        try:
            clim_names[c] = json.loads((CLIMATE_DIR / f"{c}.json").read_text(encoding="utf-8")).get("name", c)
        except Exception:
            clim_names[c] = c
    crops = available_crops()

    with st.sidebar:
        st.header("실험 설정")
        facility = st.selectbox("시설", facilities, format_func=lambda f: fac_names.get(f, f))
        crop = st.selectbox("작물", crops, index=crops.index("basil") if "basil" in crops else 0)
        climate = st.selectbox("기후", climates, index=climates.index("summer_south") if "summer_south" in climates else 0,
                               format_func=lambda c: clim_names.get(c, c))
        days = st.slider("기간(일)", 1, 14, 5)
        seed = int(st.number_input("시드", min_value=0, max_value=9999, value=0, step=1))
        controllers = st.multiselect("운전 방식", list(CONTROLLER_LABELS.keys()), default=list(DEFAULT_CONTROLLERS),
                                     format_func=_nm)

        fac_obj = load_facility(facility)
        dev_ids = [d.id for d in fac_obj.devices]
        dev_label = {d.id: d.name for d in fac_obj.devices}

        st.subheader("이벤트 (선택)")
        degrade = None
        if st.checkbox("장치 성능 저하"):
            dv = st.selectbox("저하 장치", dev_ids, format_func=lambda i: dev_label[i], key="deg_dev")
            factor = st.slider("성능 비율", 0.1, 1.0, 0.5, 0.05, key="deg_factor")
            dday = st.slider("발생일", 0, max(days - 1, 0), min(2, max(days - 1, 0)), key="deg_day") if days > 1 else 0
            degrade = (dv, float(factor), int(dday))
        remove = None
        if st.checkbox("장치 제거"):
            rv = st.selectbox("제거 장치", dev_ids, format_func=lambda i: dev_label[i], key="rm_dev")
            rday = st.slider("제거일", 0, max(days - 1, 0), min(2, max(days - 1, 0)), key="rm_day") if days > 1 else 0
            remove = (rv, int(rday))

        run = st.button("실행", type="primary", use_container_width=True)

    params = (facility, tuple(controllers), crop, climate, int(days), int(seed), degrade, remove)
    if run:
        st.session_state["params"] = params

    # 1. 장치 프로필
    st.header("1. 시설 장치 프로필")
    if fac_obj.description:
        st.caption(fac_obj.description)
    st.dataframe(device_table(facility), hide_index=True, use_container_width=True)
    st.caption("운전 방식은 장치 이름이 아니라 '기능'과 '영향 계수'만 보고 제어한다. 시설이 바뀌면 프로필만 교체한다.")

    run_params = st.session_state.get("params")
    if run_params is None:
        st.info("왼쪽에서 설정을 고르고 '실행'을 누르세요.")
        return
    if run_params != params:
        st.warning("설정이 바뀌었습니다. 아래 결과는 이전 실행 기준입니다. 다시 '실행'을 누르세요.")
    r_fac, r_ctrls, r_crop, r_clim, r_days, r_seed, r_deg, r_rm = run_params
    if not r_ctrls:
        st.warning("운전 방식을 하나 이상 선택하세요.")
        return

    with st.spinner("시뮬레이션 중..."):
        try:
            dicts = cached_run(*run_params)
        except Exception as e:  # noqa: BLE001
            st.error(f"실행 오류: {e}")
            return
    results = results_from_dicts(dicts)
    events = build_events(r_deg, r_rm)
    st.caption(
        f"시설 {fac_names.get(r_fac, r_fac)} · 작물 {r_crop} · 기후 {clim_names.get(r_clim, r_clim)} · {r_days}일 · 시드 {r_seed}"
        + (" · 이벤트: " + ", ".join(
            f"{ev.device} 성능 {ev.factor:g}배 ({ev.day}일차)" if ev.type == "degrade" else f"{ev.device} 제거 ({ev.day}일차)"
            for ev in events) if events else "")
    )

    # 2. 비교 결과
    st.header("2. 비교 결과")
    cmp = d_vs_a(results)
    if cmp is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("전기 절감률 (탄소 반영 vs 흔한 방식)", f"{cmp['energy_saving_%']:.1f}%")
        c2.metric("탄소 감축률 (탄소 반영 vs 흔한 방식)", f"{cmp['carbon_reduction_%']:.1f}%")
        c3.metric("적정 범위 유지율 (탄소 반영)", f"{cmp['in_range_D_%']:.1f}%", f"{cmp['in_range_diff_pp']:+.1f}%p vs 흔한 방식")
    else:
        st.info("비교 지표는 '흔한 방식'과 '탄소 반영'을 모두 선택해야 표시됩니다.")
    st.dataframe(summary_table(results), hide_index=True, use_container_width=True)
    with st.expander("운전 방식 설명"):
        for k in r_ctrls:
            st.markdown(f"- **{_nm(k)}**: {CONTROLLER_DESCRIPTIONS.get(k, '')}")
    if "A" not in r_ctrls:
        st.caption("절감률·감축률 열은 '흔한 방식'이 있어야 계산됩니다.")

    # 3. 그래프
    st.header("3. 시간별 변화")
    left, right = st.columns(2)
    with left:
        st.subheader("수온과 외기 온도 (°C)")
        st.line_chart(water_temp_frame(results))
    with right:
        st.subheader("누적 탄소 배출 (kgCO2e)")
        st.line_chart(cumulative_carbon_frame(results))
    left, right = st.columns(2)
    with left:
        st.subheader("시간대별 전력 배출계수")
        st.bar_chart(grid_ef_frame())
        st.caption("D는 배출계수가 낮은 시간대로 전력 사용을 옮긴다.")
    with right:
        st.subheader("수온 예측오차 (24시간 이동평균, °C)")
        pe = pred_error_frame(results)
        if pe.empty:
            st.info("'학습형' 또는 '탄소 반영'을 선택하면 표시됩니다.")
        else:
            st.line_chart(pe)
            st.caption("시간이 지날수록 오차가 줄면 모델이 이 시설에 적응한 것이다.")

    # 4. 학습 계수
    st.header("4. 학습된 장치 영향 계수")
    ct = coefficient_table(results, r_fac)
    if ct.empty:
        st.info("'학습형' 또는 '탄소 반영'을 선택하면 표시됩니다.")
    else:
        st.dataframe(ct, hide_index=True, use_container_width=True)
        st.caption("사전값은 프로필에 적힌 추정치, 학습값은 운전 데이터로 보정한 값이다 (시간당 변화량 / 제어 단위). 학습/사전이 1보다 작으면 실제 효과가 예상보다 약하다는 뜻이다.")

    # 5. 장치 명령
    st.header("5. 장치 명령 예시")
    who, cmds = last_commands(results, r_fac, events)
    if not cmds:
        st.info("표시할 명령이 없습니다.")
    else:
        st.caption(f"{_nm(who) if who else ''} 방식의 마지막 시간 제어값을 하드웨어 명령으로 변환한 결과")
        st.json(cmds)


if __name__ == "__main__":
    main()
