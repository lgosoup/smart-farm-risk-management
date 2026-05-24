from __future__ import annotations

from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from control import (  # noqa: E402,F401
    CONTROL_KEYS,
    DANGER,
    EMERGENCY,
    MONITOR_ONLY,
    OPTIMIZE,
    WARNING,
    build_device_commands,
    build_monitor_only_output,
    calculate_change_penalty,
    calculate_device_energy_breakdown,
    calculate_energy_wh,
    calculate_over_control_penalty,
    calculate_resource_cost,
    calculate_weak_reduction_penalty,
    candidate_brief,
    candidate_max_value,
    clamp,
    control_option_key_map,
    evaluate_candidate,
    generate_candidates,
    get_default_optimizer_config,
    is_valid_candidate,
    load_assessment_context,
    load_previous_control,
    merged_config,
    now_iso,
    optimize_control,
    predict_next_inputs,
    raw_candidate_count,
    read_jsonl,
    risk_label,
    risk_value,
    round_numeric_dict,
    run_demo_scenarios,
    save_control_output,
    should_optimize,
    trim_jsonl_keep_last,
    zero_control,
)

_calculate_device_energy_breakdown = calculate_device_energy_breakdown
_candidate_brief = candidate_brief
_candidate_max_value = candidate_max_value
_control_option_key_map = control_option_key_map
_load_assessment_context = load_assessment_context
_merged_config = merged_config
_now_iso = now_iso
_raw_candidate_count = raw_candidate_count
_read_jsonl = read_jsonl
_risk_label = risk_label
_risk_value = risk_value
_round_numeric_dict = round_numeric_dict
_zero_control = zero_control
