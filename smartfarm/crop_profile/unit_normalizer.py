"""Unit normalization for crop parameter candidates."""
from __future__ import annotations

from typing import Any


# Standard units used by the fuzzy system
STANDARD_UNITS: dict[str, str] = {
    "temperature": "℃",      # ℃
    "ph": "pH",                   # dimensionless
    "ec": "mS/cm",
    "do": "mg/L",
    "fr": "ratio",                # dimensionless
    "turbidity": "NTU",
}


class UnitNormalizer:
    """Normalize measurement units to the fuzzy system's standard units.

    Each candidate dict is expected to have at minimum:
        - ``param``: str   (e.g. "ph_optimal", "ec_vegetative", "air_temp_optimal")
        - ``unit``: str    (original unit string)
        - ``value``: float | list[float, float]  (scalar or [min, max] range)

    The method returns a new list of dicts, each augmented with:
        - ``unit_original``    : the original unit string
        - ``value_normalized`` : scalar float or [min, max] list in standard units
        - ``unit_normalized``  : the standard unit string for this parameter
        - ``unit_conversion``  : {"converted": bool, "rule": str}
    """

    # ------------------------------------------------------------------ #
    # Internal lookup: map param keywords → standard unit key             #
    # ------------------------------------------------------------------ #
    _PARAM_TO_UNIT_KEY: dict[str, str] = {
        "temp": "temperature",
        "ph": "ph",
        "ec": "ec",
        "do": "do",
        "flow": "fr",
        "fr": "fr",
        "turbidity": "turbidity",
    }

    def _resolve_unit_key(self, param: str) -> str | None:
        param_lower = param.lower()
        for kw, uk in self._PARAM_TO_UNIT_KEY.items():
            if kw in param_lower:
                return uk
        return None

    # ------------------------------------------------------------------ #
    # Value helpers                                                        #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _apply_factor(value: float | list, factor: float) -> float | list:
        if isinstance(value, list):
            return [round(v * factor, 6) for v in value]
        return round(float(value) * factor, 6)

    @staticmethod
    def _apply_f_to_c(value: float | list) -> float | list:
        def _conv(f: float) -> float:
            return round((f - 32) * 5.0 / 9.0, 4)
        if isinstance(value, list):
            return [_conv(v) for v in value]
        return _conv(float(value))

    # ------------------------------------------------------------------ #
    # Per-unit conversion logic                                            #
    # ------------------------------------------------------------------ #
    def _convert_ec(
        self, unit: str, value: float | list, ppm_scale: int | None
    ) -> tuple[float | list, str, bool]:
        unit_lower = unit.lower().replace(" ", "")
        if unit_lower in ("ms/cm", "ms/cm"):
            return value, "1:1 passthrough (already mS/cm)", True
        if unit_lower in ("ds/m", "ds/m"):
            # 1 dS/m == 1 mS/cm
            return value, "dS/m → mS/cm ×1.0", True
        if unit_lower in ("µs/cm", "us/cm", "µs/cm"):
            return self._apply_factor(value, 1.0 / 1000.0), "µS/cm ÷ 1000 → mS/cm", True
        if "ppm" in unit_lower:
            if ppm_scale == 500:
                return self._apply_factor(value, 1.0 / 500.0), "ppm ÷ 500 → mS/cm (scale=500)", True
            if ppm_scale == 700:
                return self._apply_factor(value, 1.0 / 700.0), "ppm ÷ 700 → mS/cm (scale=700)", True
            return value, "ppm: scale unknown, manual review needed", False
        # Unknown EC unit
        return value, f"unknown EC unit '{unit}', no conversion", False

    def _convert_temperature(
        self, unit: str, value: float | list
    ) -> tuple[float | list, str, bool]:
        unit_stripped = unit.strip().replace("°", "").upper()
        if unit_stripped in ("℃", "C", "°C", "DEGC"):
            return value, "1:1 passthrough (already ℃)", True
        if unit_stripped in ("F", "°F", "DEGF"):
            return self._apply_f_to_c(value), "°F → ℃ (F-32)×5/9", True
        return value, f"unknown temperature unit '{unit}', no conversion", False

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #
    def normalize(self, candidates: list[dict]) -> list[dict]:
        """Normalize units for all candidates.

        Returns a new list; each element is a shallow copy of the input dict
        augmented with the four normalization fields.
        """
        result: list[dict] = []
        for cand in candidates:
            c = dict(cand)  # shallow copy
            param = c.get("param", "")
            unit: str = c.get("unit", "")
            value: Any = c.get("value")
            ppm_scale: int | None = c.get("ppm_scale", None)

            c["unit_original"] = unit

            unit_key = self._resolve_unit_key(param)

            if unit_key is None:
                # Cannot determine what this parameter maps to
                c["value_normalized"] = value
                c["unit_normalized"] = unit or "unknown"
                c["unit_conversion"] = {"converted": False, "rule": "param type unrecognized"}
                result.append(c)
                continue

            standard_unit = STANDARD_UNITS[unit_key]
            c["unit_normalized"] = standard_unit

            converted_value = value
            rule = ""
            converted = True

            if unit_key == "temperature":
                converted_value, rule, converted = self._convert_temperature(unit, value)

            elif unit_key == "ec":
                converted_value, rule, converted = self._convert_ec(unit, value, ppm_scale)

            elif unit_key == "ph":
                # pH is dimensionless; just pass through
                rule = "pH dimensionless passthrough"
                converted = True

            elif unit_key == "do":
                unit_lower = unit.lower()
                if unit_lower in ("mg/l", "mg/l"):
                    rule = "1:1 passthrough (already mg/L)"
                else:
                    rule = f"unknown DO unit '{unit}', no conversion"
                    converted = False

            elif unit_key == "fr":
                rule = "flow ratio dimensionless passthrough"
                converted = True

            elif unit_key == "turbidity":
                unit_lower = unit.lower()
                if unit_lower == "ntu":
                    rule = "1:1 passthrough (already NTU)"
                elif unit_lower == "fnu":
                    # FNU ≈ NTU for most practical purposes
                    rule = "FNU treated as NTU (1:1 approximate)"
                elif unit_lower == "lux":
                    rule = "lux → NTU conversion requires light-source spec; manual review"
                    converted = False
                else:
                    rule = f"unknown turbidity unit '{unit}', no conversion"
                    converted = False

            c["value_normalized"] = converted_value
            c["unit_conversion"] = {"converted": converted, "rule": rule}
            result.append(c)

        return result
