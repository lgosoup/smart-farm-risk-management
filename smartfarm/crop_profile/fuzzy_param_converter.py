"""Convert raw crop growth parameters to fuzzy membership point overrides."""
from __future__ import annotations


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _clamp_list(pts: list[float], lo: float, hi: float) -> list[float]:
    return [_clamp(v, lo, hi) for v in pts]


_T_MIN, _T_MAX = -10.0, 60.0
_PH_MIN, _PH_MAX = 0.0, 14.5
_EC_MIN, _EC_MAX = 0.0, 20.0


class FuzzyParamConverter:
    """Convert raw crop growth parameters to fuzzy membership point overrides.

    ``raw_growth_params`` keys (all optional):
        - ``air_temp_optimal``   : [min_C, max_C] or None
        - ``water_temp_optimal`` : [min_C, max_C] or None  (수온 우선, 없으면 공기 온도 사용)
        - ``ph_optimal``         : [min_pH, max_pH] or None
        - ``ec_vegetative``      : [min_mS, max_mS] or None

    Returns a dict with keys water_temperature, pH, EC_hydro.
    """

    def _convert_temperature(self, opt_min: float, opt_max: float) -> dict:
        lo, hi = _T_MIN, _T_MAX
        stable   = _clamp_list([opt_min - 1, opt_min, opt_max, opt_max + 1], lo, hi)
        border   = _clamp_list([opt_max + 0.5, opt_max + 3.5, opt_max + 6.5], lo, hi)
        warning  = _clamp_list([opt_max + 5, opt_max + 9, opt_max + 13], lo, hi)
        danger   = _clamp_list([opt_max + 11, opt_max + 15, opt_max + 28, opt_max + 33], lo, hi)
        low      = _clamp_list([opt_min - 7, opt_min - 4, opt_min - 1], lo, hi)
        very_low = _clamp_list([opt_min - 22, opt_min - 17, opt_min - 6, opt_min - 2], lo, hi)
        return {
            "매우낮음":    very_low,
            "낮음":        low,
            "안정(최적)":  stable,
            "경계(상승)":  border,
            "경고(고온)":  warning,
            "위험(급고온)": danger,
        }

    def _convert_ph(self, ph_min: float, ph_max: float) -> dict:
        lo, hi = _PH_MIN, _PH_MAX
        span = ph_max - ph_min
        if span < 0.5:
            stable = _clamp_list([ph_min - 0.1, ph_min, ph_max, ph_max + 0.1], lo, hi)
        else:
            stable = _clamp_list([ph_min, ph_min + 0.2, ph_max - 0.2, ph_max], lo, hi)
        mild_acid   = _clamp_list([ph_min - 1.0, ph_min - 0.5, ph_min + 0.2], lo, hi)
        strong_acid = _clamp_list([0.0, 0.0, ph_min - 1.2, ph_min - 0.6], lo, hi)
        mild_alk    = _clamp_list([ph_max - 0.2, ph_max + 0.5, ph_max + 0.9], lo, hi)
        strong_alk  = _clamp_list([ph_max + 0.7, ph_max + 1.2, 14.0, 14.5], lo, hi)
        return {
            "강산성(위험)":       strong_acid,
            "약산성(경고–주의)":  mild_acid,
            "안정(최적)":         stable,
            "약알칼리(경고–주의)": mild_alk,
            "강알칼리(위험)":     strong_alk,
        }

    def _convert_ec_hydro(self, ec_min: float, ec_max: float) -> dict:
        lo, hi = _EC_MIN, _EC_MAX
        stable    = _clamp_list([ec_min, ec_min + 0.1, ec_max - 0.1, ec_max], lo, hi)
        very_low  = _clamp_list([0.0, 0.0, ec_min * 0.3, ec_min * 0.5], lo, hi)
        low       = _clamp_list([ec_min * 0.4, ec_min * 0.7, ec_min], lo, hi)
        high_warn = _clamp_list([ec_max, ec_max + 0.4, ec_max + 0.8], lo, hi)
        very_high = _clamp_list([ec_max + 0.6, ec_max + 1.2, ec_max + 3.0, ec_max + 4.0], lo, hi)
        return {
            "매우낮음":      very_low,
            "낮음":          low,
            "안정":          stable,
            "높음(경고)":    high_warn,
            "매우높음(위험)": very_high,
        }

    def convert(
        self,
        raw_growth_params: dict,
        fallback_defaults: dict | None = None,
    ) -> dict:
        """Convert raw growth parameters to fuzzy membership point overrides.

        Returns dict with keys: water_temperature, pH, EC_hydro
        (water_temperature는 water_temp_optimal 우선, 없으면 air_temp_optimal 사용)
        """
        fb = fallback_defaults or {}

        def _get(key: str):
            v = raw_growth_params.get(key)
            if v is None:
                v = fb.get(key)
            return v

        result: dict = {}

        # ── Water temperature (수온 우선, 없으면 공기 온도 사용) ────────────
        t_opt = _get("water_temp_optimal") or _get("air_temp_optimal")
        if isinstance(t_opt, (list, tuple)) and len(t_opt) == 2:
            result["water_temperature"] = self._convert_temperature(float(t_opt[0]), float(t_opt[1]))

        # ── pH ───────────────────────────────────────────────────────────────
        ph_opt = _get("ph_optimal")
        if isinstance(ph_opt, (list, tuple)) and len(ph_opt) == 2:
            result["pH"] = self._convert_ph(float(ph_opt[0]), float(ph_opt[1]))

        # ── EC_hydro ─────────────────────────────────────────────────────────
        ec_veg = _get("ec_vegetative")
        if isinstance(ec_veg, (list, tuple)) and len(ec_veg) == 2:
            result["EC_hydro"] = self._convert_ec_hydro(float(ec_veg[0]), float(ec_veg[1]))

        return result
