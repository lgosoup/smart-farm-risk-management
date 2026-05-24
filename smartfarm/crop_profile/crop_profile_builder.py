"""Assemble a complete Crop Profile JSON from validated extraction candidates."""
from __future__ import annotations

from .unit_normalizer import UnitNormalizer
from .candidate_validator import CandidateValidator
from .fuzzy_param_converter import FuzzyParamConverter


# ── Default fallback values used when candidate data is missing ───────────────
_FALLBACK_GROWTH_PARAMS: dict = {
    "air_temp_optimal":   [20.0, 25.0],
    "water_temp_optimal": None,
    "ph_optimal":         [6.0, 7.0],
    "ec_vegetative":      [1.0, 2.0],
    "do_minimum":         5.0,
}

_DEFAULT_RULE_WEIGHTS: dict = {
    "R6_high_temp_low_do": 1.0,
    "R7_low_flow_low_do":  1.0,
    "R8_low_flow_turbidity": 1.0,
    "R9_high_ec_low_do":   1.0,
}

_DEFAULT_IMPROVEMENT_LIMITS: dict = {
    "T": 0.5, "DO": 0.3, "pH": 0.15,
    "FR": 0.08, "EC": 0.25, "Turbidity": 1.5,
}

_DEFAULT_SYSTEM_FLAGS: dict = {
    "root_oxygen_sensitivity": "medium",
    "root_disease_sensitivity": "medium",
    "heat_sensitivity": "medium",
    "extra_rules": [],
}


def _midpoint(lo: float, hi: float) -> float:
    return round((lo + hi) / 2.0, 3)


def _extract_raw_params(passed_candidates: list[dict]) -> tuple[dict, list[str]]:
    """Build raw_growth_params dict from passed candidates.

    Returns (raw_params, fallbacks_used).
    """
    raw: dict = {
        "air_temp_optimal":   None,
        "water_temp_optimal": None,
        "ph_optimal":         None,
        "ec_vegetative":      None,
        "do_minimum":         None,
    }
    fallbacks_used: list[str] = []

    # Map candidate param names → raw_growth_params keys
    param_map: dict[str, str] = {
        "air_temp_optimal":   "air_temp_optimal",
        "water_temp_optimal": "water_temp_optimal",
        "ph_optimal":         "ph_optimal",
        "ec_vegetative":      "ec_vegetative",
        "do_minimum":         "do_minimum",
    }
    # Also handle aliases
    aliases: dict[str, str] = {
        "t_air_optimal": "air_temp_optimal",
        "t_water_optimal": "water_temp_optimal",
        "ph": "ph_optimal",
        "ec": "ec_vegetative",
        "do_min": "do_minimum",
    }

    for c in passed_candidates:
        param = c.get("param", "").lower().strip()
        target = param_map.get(param) or aliases.get(param)
        if target and raw.get(target) is None:
            raw[target] = c.get("value_normalized") or c.get("value")

    # Fill any still-None with global fallbacks
    for key, fallback_val in _FALLBACK_GROWTH_PARAMS.items():
        if raw.get(key) is None:
            raw[key] = fallback_val
            if fallback_val is not None:
                fallbacks_used.append(key)

    return raw, fallbacks_used


def _build_control_targets(raw: dict) -> dict:
    """Compute control targets as midpoints of optimal ranges."""

    def mid(v) -> float | None:
        if v is None:
            return None
        if isinstance(v, (list, tuple)) and len(v) == 2:
            return _midpoint(float(v[0]), float(v[1]))
        return float(v)

    t_air = mid(raw.get("air_temp_optimal"))
    t_water = mid(raw.get("water_temp_optimal"))
    target_temp = t_water if t_water is not None else t_air

    return {
        "target_temp": target_temp,
        "target_ph": mid(raw.get("ph_optimal")),
        "target_ec": mid(raw.get("ec_vegetative")),
        "target_do": raw.get("do_minimum"),  # minimum, not midpoint
        "target_flow_ratio": 1.0,
        "ec_mode": "hydro",
    }


def _build_sources(passed_candidates: list[dict]) -> dict:
    """Group passed candidates by param field into sources dict."""
    sources: dict[str, list[dict]] = {}
    for c in passed_candidates:
        param = c.get("param", "unknown")
        record = {
            "source": c.get("source", ""),
            "evidence": c.get("evidence", ""),
            "confidence": c.get("confidence"),
            "context": c.get("context", ""),
        }
        sources.setdefault(param, []).append(record)
    return sources


class CropProfileBuilder:
    """Assemble the full Crop Profile JSON from all components.

    Usage::

        builder = CropProfileBuilder()
        profile = builder.build(
            crop_info={"crop": "basil", "crop_ko": "바질", ...},
            candidates=[...],
        )
    """

    def __init__(self) -> None:
        self.unit_normalizer = UnitNormalizer()
        self.validator = CandidateValidator()
        self.fuzzy_converter = FuzzyParamConverter()

    def build(
        self,
        crop_info: dict,
        candidates: list[dict],
        sources_metadata: list[dict] | None = None,  # noqa: ARG002
        system_flags_override: dict | None = None,
    ) -> dict:
        """Build complete Crop Profile from validated extraction candidates.

        Parameters
        ----------
        crop_info:
            Metadata about the crop (name, growth_type, etc.).  Expected keys:
            crop, crop_ko, scientific_name, cultivar, growth_type, edible_part,
            cultivation_mode.
        candidates:
            Raw extraction candidates (before or after unit normalization).
        sources_metadata:
            Optional list of additional source records to attach.
        system_flags_override:
            Override the system_flags sub-dict.
        """
        # 1. Normalise units
        normalized = self.unit_normalizer.normalize(candidates)

        # 2. Validate
        validation_result = self.validator.validate(normalized, crop_info)
        passed = validation_result["passed"]

        # 3. Extract raw growth params from passed candidates
        raw_growth_params, fallbacks_used = _extract_raw_params(passed)

        # 4. Generate fuzzy params
        fuzzy_params = self.fuzzy_converter.convert(raw_growth_params)

        # 5. Control targets
        control_targets = _build_control_targets(raw_growth_params)

        # 6. Rule weights (merge crop_info overrides if provided)
        rule_weights = dict(_DEFAULT_RULE_WEIGHTS)
        for key, val in crop_info.get("rule_weights_override", {}).items():
            rule_key = f"{key}_high_temp_low_do" if key == "R6" else key
            rule_weights[rule_key] = val

        # 7. Improvement limits (crop_info can override per-variable caps)
        improvement_limits = dict(_DEFAULT_IMPROVEMENT_LIMITS)
        improvement_limits.update(crop_info.get("improvement_limits_override", {}))

        # 8. System flags
        system_flags = dict(_DEFAULT_SYSTEM_FLAGS)
        system_flags.update(crop_info.get("system_flags", {}))
        if system_flags_override:
            system_flags.update(system_flags_override)

        # 9. Sources dict
        sources = _build_sources(passed)

        # 10. Validation summary
        validation_summary = {
            "passed_count": len(passed),
            "pending_count": len(validation_result["pending"]),
            "rejected_count": len(validation_result["rejected"]),
            "overall_confidence": crop_info.get("overall_confidence", 0.70),
            "pending": [
                {"param": c.get("param"), "reason": c.get("pending_reason")}
                for c in validation_result["pending"]
            ],
            "rejected": [
                {"param": c.get("param"), "reason": c.get("reject_reason")}
                for c in validation_result["rejected"]
            ],
        }

        profile: dict = {
            "crop":              crop_info.get("crop", ""),
            "crop_ko":           crop_info.get("crop_ko", ""),
            "scientific_name":   crop_info.get("scientific_name"),
            "cultivar":          crop_info.get("cultivar"),
            "growth_type":       crop_info.get("growth_type", ""),
            "edible_part":       crop_info.get("edible_part"),
            "cultivation_mode":  crop_info.get("cultivation_mode", "hydro"),

            "raw_growth_params": raw_growth_params,
            "fuzzy_params":      fuzzy_params,
            "control_targets":   control_targets,
            "rule_weights":      rule_weights,
            "improvement_limits": improvement_limits,
            "system_flags":      system_flags,
            "sources":           sources,
            "fallbacks_used":    fallbacks_used,
            "validation":        validation_summary,
        }

        return profile
