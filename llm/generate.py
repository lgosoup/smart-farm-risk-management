# llm/generate.py
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import Counter

from .loader import get_llm
from .prompts import build_system_prompt

_CROP_CONFIG_PATH = Path(__file__).resolve().parent.parent / "data" / "crop_config.json"


def _load_system_prompt() -> str:
    """data/crop_config.json에서 작물명을 읽어 시스템 프롬프트를 생성한다."""
    try:
        data = json.loads(_CROP_CONFIG_PATH.read_text(encoding="utf-8"))
        crop_name = data.get("crop", "")
        crop_ko = data.get("crop_ko", "")
        return build_system_prompt(crop_name=crop_name, crop_ko=crop_ko)
    except Exception:
        return build_system_prompt()


def _read_jsonl_tail(path: str, n: int) -> List[Dict[str, Any]]:
    if n <= 0:
        return []
    p = Path(path)
    if not p.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # 마지막 줄이 잘려있거나 깨진 경우 스킵(네 샘플 케이스 대응)
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows[-n:]


def _strip_think_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_first_json_object(text: str) -> str:
    t = _strip_think_blocks(text).strip()
    t = re.sub(r"^\s*```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```\s*$", "", t)

    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output")
    return t[start : end + 1]


def _postprocess_korean_terms(obj: Any) -> Any:
    """
    최소 변경 후처리:
    - 출력 JSON의 "값 문자열"에서만 EC/DO 약어를 한국어로 풀어씀
    - 이미 (EC), (DO) 형태로 들어간 건 중복 치환 방지
    - 키 이름은 절대 변경하지 않음(값만 변경)
    """
    # "(EC)" 안의 EC는 건드리지 않기 위해 (?<!\() 사용
    repls = [
        (re.compile(r"(?<!\()EC\b"), "전기전도도(EC)"),
        (re.compile(r"(?<!\()DO\b"), "용존산소(DO)"),
    ]

    if isinstance(obj, str):
        s = obj
        for pat, rep in repls:
            s = pat.sub(rep, s)
        return s
    if isinstance(obj, list):
        return [_postprocess_korean_terms(x) for x in obj]
    if isinstance(obj, dict):
        # 키는 그대로 두고 값만 후처리
        return {k: _postprocess_korean_terms(v) for k, v in obj.items()}
    return obj


def _ensure_schema_json(text: str) -> str:
    """
    개선 3: JSON-only 강건화
    - think/코드블록 제거
    - 첫 { ~ 마지막 } 추출
    - json 파싱 시도
    - trailing comma 같은 흔한 깨짐 보정 후 재파싱
    """
    extracted = _extract_first_json_object(text)
    try:
        obj = json.loads(extracted)
    except json.JSONDecodeError:
        cleaned = re.sub(r",\s*([}\]])", r"\1", extracted)
        obj = json.loads(cleaned)

    # ✅ (딱 이거만) EC/DO를 한국어로 풀어쓴 표기로 후처리
    obj = _postprocess_korean_terms(obj)

    return json.dumps(obj, ensure_ascii=False)


def _safe_inputs(packet: Dict[str, Any]) -> Dict[str, Any]:
    v = packet.get("inputs")
    return v if isinstance(v, dict) else {}


def _safe_list(d: Dict[str, Any], key: str) -> List[Any]:
    v = d.get(key)
    return v if isinstance(v, list) else []


def _safe_dict(d: Dict[str, Any], key: str) -> Dict[str, Any]:
    v = d.get(key)
    return v if isinstance(v, dict) else {}


def _compute_max_delta(history_packets: List[Dict[str, Any]]) -> Dict[str, float]:
    # consecutive inputs delta 기반 최대 변화량
    vars_ = ["T", "DO", "pH", "FR", "EC", "Turbidity"]
    max_delta: Dict[str, float] = {k: 0.0 for k in vars_}

    prev_inputs: Optional[Dict[str, Any]] = None
    for p in history_packets:
        cur_inputs = _safe_inputs(p)
        if not cur_inputs:
            continue
        if prev_inputs is not None:
            for k in vars_:
                a = prev_inputs.get(k)
                b = cur_inputs.get(k)
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    d = abs(b - a)
                    if d > max_delta[k]:
                        max_delta[k] = float(d)
        prev_inputs = cur_inputs

    # sensor_integrity.spike에서 delta가 있으면 그것도 반영(더 명시적인 급변)
    for p in history_packets:
        spike = _safe_dict(_safe_dict(p, "sensor_integrity"), "spike")
        for k, info in spike.items():
            if isinstance(info, dict):
                d = info.get("delta")
                if isinstance(d, (int, float)) and k in max_delta:
                    if abs(d) > max_delta[k]:
                        max_delta[k] = float(abs(d))

    return max_delta


def _compute_label_stats(history_packets: List[Dict[str, Any]]) -> Dict[str, Any]:
    labels = []
    for p in history_packets:
        ak = p.get("action_key")
        if isinstance(ak, str) and ak:
            labels.append(ak)

    counts = dict(Counter(labels))

    # longest streak
    longest_label = None
    longest_len = 0
    cur_label = None
    cur_len = 0
    for lab in labels:
        if lab == cur_label:
            cur_len += 1
        else:
            cur_label = lab
            cur_len = 1
        if cur_len > longest_len:
            longest_len = cur_len
            longest_label = cur_label

    recent_transition = None
    if len(labels) >= 2 and labels[-2] != labels[-1]:
        recent_transition = f"{labels[-2]}→{labels[-1]}"

    return {
        "label_counts": counts,
        "longest_streak": {"label": longest_label, "len": longest_len} if longest_label else {"label": None, "len": 0},
        "recent_transition": recent_transition,
    }


def _compute_flag_stats(history_packets: List[Dict[str, Any]]) -> Dict[str, Any]:
    flag_counter = Counter()
    for p in history_packets:
        flags = _safe_list(_safe_dict(p, "timeseries"), "flags")
        for f in flags:
            if isinstance(f, str) and f:
                flag_counter[f] += 1
    top = [k for k, _ in flag_counter.most_common(5)]
    return {"flags_top": top, "flags_counts": dict(flag_counter)}


def _build_ts_summary(history_packets: List[Dict[str, Any]], window: int) -> Dict[str, Any]:
    hp = history_packets[-window:] if window > 0 else history_packets
    label_stats = _compute_label_stats(hp)
    flag_stats = _compute_flag_stats(hp)
    max_delta = _compute_max_delta(hp)

    # max_delta에서 가장 큰 변수 1개도 같이 제공(설명에 쓰기 쉬움)
    max_var = None
    max_val = 0.0
    for k, v in max_delta.items():
        if v > max_val:
            max_val = v
            max_var = k

    return {
        "window": f"최근 {len(hp)}개",
        **label_stats,
        **flag_stats,
        "max_delta": max_delta,
        "max_delta_top": {"var": max_var, "delta": max_val},
    }


def _compact_history_packets(history_packets: List[Dict[str, Any]], max_hist_packets: int) -> List[Dict[str, Any]]:
    """
    원본 history_packets는 크니, LLM에 넣을 건 요약본만.
    """
    tail = history_packets[-max_hist_packets:]
    compact: List[Dict[str, Any]] = []

    for p in tail:
        inputs = _safe_inputs(p)
        risk = _safe_dict(p, "risk")
        ts = _safe_dict(p, "timeseries")
        integ = _safe_dict(p, "sensor_integrity")

        compact.append(
            {
                "timestamp": inputs.get("timestamp"),
                "action_key": p.get("action_key"),
                "risk_score": risk.get("risk_score"),
                "risk_label": risk.get("risk_label"),
                "flags": _safe_list(ts, "flags"),
                "spike": _safe_dict(integ, "spike"),
                "out_of_range": _safe_dict(integ, "out_of_range"),
            }
        )
    return compact


def _build_payload(
    *,
    current_packet: Dict[str, Any],
    history_packets: Optional[List[Dict[str, Any]]] = None,
    current_input: Optional[Dict[str, Any]] = None,
    user_query: Optional[str] = None,
    ts_window: int = 12,
    max_hist_packets: int = 8,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"current_packet": current_packet}

    # current_packet에 inputs 없을 때만 current_input 보조
    packet_has_inputs = isinstance(current_packet.get("inputs"), dict)
    if (not packet_has_inputs) and isinstance(current_input, dict):
        payload["current_input"] = current_input

    if isinstance(user_query, str) and user_query.strip():
        payload["user_query"] = user_query.strip()

    if history_packets:
        payload["ts_summary"] = _build_ts_summary(history_packets, window=ts_window)
        payload["history_packets"] = _compact_history_packets(history_packets, max_hist_packets=max_hist_packets)

    return payload


def run_smartfarm_llm(payload: Dict[str, Any]) -> str:
    tokenizer, model = get_llm()
    system_prompt = _load_system_prompt()

    payload_str = json.dumps(payload, ensure_ascii=False, indent=2)
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "/no_think\n"
                "아래 JSON만 근거로, 출력 스키마에 맞춰 JSON object 1개만 출력해. "
                "설명 문장, 코드블록, <think> 태그를 출력하지 마.\n\n"
                f"{payload_str}"
            ),
        },
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda")

    # ✅ do_sample=False면 temperature/top_p/top_k는 제거(경고 방지)
    out = model.generate(
        **inputs,
        max_new_tokens=360,
        do_sample=False,
    )

    raw = tokenizer.decode(
        out[0][inputs["input_ids"].shape[-1] :],
        skip_special_tokens=True,
    ).strip()

    try:
        return _ensure_schema_json(raw)
    except Exception:
        # 1회 재시도
        retry_messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "너의 이전 출력은 JSON 파싱에 실패했다. "
                    "출력은 반드시 JSON object 1개만. "
                    "코드블록/설명/<think> 금지.\n\n"
                    f"{payload_str}"
                ),
            },
        ]
        retry_inputs = tokenizer.apply_chat_template(
            retry_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to("cuda")

        retry_out = model.generate(
            **retry_inputs,
            max_new_tokens=360,
            do_sample=False,
        )

        retry_raw = tokenizer.decode(
            retry_out[0][retry_inputs["input_ids"].shape[-1] :],
            skip_special_tokens=True,
        ).strip()

        return _ensure_schema_json(retry_raw)


def run_smartfarm_llm_timeseries(
    *,
    current_packet: Dict[str, Any],
    history_packets: List[Dict[str, Any]],
    current_input: Optional[Dict[str, Any]] = None,
    user_query: Optional[str] = None,
) -> str:
    payload = _build_payload(
        current_packet=current_packet,
        history_packets=history_packets,
        current_input=current_input,
        user_query=user_query,
    )
    return run_smartfarm_llm(payload)
