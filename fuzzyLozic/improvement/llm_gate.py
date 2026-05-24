from __future__ import annotations

import json
import re
from typing import Any, Dict, List


def _strip_think_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_first_json_object(text: str) -> Dict[str, Any]:
    t = _strip_think_blocks(text).strip()
    t = re.sub(r"^\s*```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```\s*$", "", t)
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found")
    raw = t[start : end + 1]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raw2 = re.sub(r",\s*([}\]])", r"\1", raw)
        return json.loads(raw2)


def heuristic_gate(
    *,
    current_packet: Dict[str, Any],
    recent_packets: List[Dict[str, Any]],
    action: Dict[str, Any],
) -> Dict[str, Any]:
    """Fallback gate when LLM is unavailable.

    Conservative rules:
    - If integrity spike/out_of_range exists in current, HOLD.
    - If action is membership shift and current is already unstable (many TS flags), HOLD.
    - Otherwise, APPROVE.
    """
    integ = current_packet.get("sensor_integrity") if isinstance(current_packet.get("sensor_integrity"), dict) else {}
    spk = integ.get("spike") if isinstance(integ.get("spike"), dict) else {}
    oor = integ.get("out_of_range") if isinstance(integ.get("out_of_range"), dict) else {}

    ts = current_packet.get("timeseries") if isinstance(current_packet.get("timeseries"), dict) else {}
    flags = ts.get("flags") if isinstance(ts.get("flags"), list) else []

    bias_tags: List[str] = []
    notes: List[str] = []

    if spk or oor:
        bias_tags.append("NOISE_SUSPECT")
        notes.append("현재 패킷에 spike/out_of_range가 있어 개선 적용을 보류")
        return {"decision": "HOLD", "reason": "노이즈 의심", "bias_tags": bias_tags, "notes": notes}

    # if many warnings/dangers recently, prefer stability
    recent_labels: List[str] = []
    for p in recent_packets[-10:]:
        ak = p.get("action_key")
        if isinstance(ak, str):
            recent_labels.append(ak)
    danger_like = sum(1 for x in recent_labels if x in {"경고", "위험", "긴급"})
    if danger_like >= 7:
        bias_tags.append("STABILITY_NEED")
        notes.append("최근 경고/위험 비중이 높음")

    if isinstance(action, dict) and action.get("action_type") == "membership_points" and len(flags) >= 3:
        bias_tags.append("STABILITY_NEED")
        notes.append("시계열 플래그가 많아 멤버십 변경은 보수적으로")
        return {"decision": "HOLD", "reason": "상태 불안정", "bias_tags": bias_tags, "notes": notes}

    return {"decision": "APPROVE", "reason": "안전 범위 내 미세 조정", "bias_tags": bias_tags, "notes": notes}


def llm_gate(
    *,
    current_packet: Dict[str, Any],
    recent_packets: List[Dict[str, Any]],
    action: Dict[str, Any],
) -> Dict[str, Any]:
    """LLM gate. If model load fails, fall back to heuristic_gate."""
    try:
        from llm.loader import get_llm
        from llm.improvement_prompts import SYSTEM_PROMPT_IMPROVEMENT_GATE

        tokenizer, model = get_llm()

        payload = {
            "current_packet": current_packet,
            "recent_packets": recent_packets[-8:],
            "action": action,
        }
        payload_str = json.dumps(payload, ensure_ascii=False, indent=2)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_IMPROVEMENT_GATE},
            {
                "role": "user",
                "content": "/no_think\n아래 JSON만 근거로 스키마대로 JSON object 1개만 출력.\n\n" + payload_str,
            },
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to("cuda")

        out = model.generate(
            **inputs,
            max_new_tokens=220,
            do_sample=False,
        )

        raw = tokenizer.decode(
            out[0][inputs["input_ids"].shape[-1] :],
            skip_special_tokens=True,
        ).strip()

        obj = _extract_first_json_object(raw)
        if not isinstance(obj, dict):
            raise ValueError("gate output not a dict")

        # Hard clamp: allow only APPROVE/HOLD
        dec = obj.get("decision")
        if dec not in {"APPROVE", "HOLD"}:
            obj["decision"] = "HOLD"
            obj["reason"] = "형식 불일치로 보류"
        return obj
    except Exception:
        return heuristic_gate(current_packet=current_packet, recent_packets=recent_packets, action=action)
