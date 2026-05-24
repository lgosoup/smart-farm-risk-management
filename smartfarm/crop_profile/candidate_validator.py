"""Candidate validation for crop profile extraction results."""
from __future__ import annotations


# Physical sanity bounds per parameter
_PHYSICAL_BOUNDS: dict[str, tuple[float, float]] = {
    "ph":   (0.0, 14.0),
    "ec":   (0.0, 20.0),   # mS/cm
    "temp": (-10.0, 60.0), # ℃ (air)
    "do":   (0.0, 20.0),   # mg/L
}

_CONFIDENCE_PASS_THRESHOLD = 0.65
_CONFIDENCE_PENDING_LOWER = 0.40


def _param_key(param: str) -> str | None:
    """Map a param string to a physical-bounds category."""
    p = param.lower()
    if "ph" in p:
        return "ph"
    if "ec" in p:
        return "ec"
    if "temp" in p:
        return "temp"
    if "do" in p:
        return "do"
    return None


def _get_range(candidate: dict) -> tuple[float | None, float | None]:
    """Extract (min, max) from value_normalized; supports scalar and list."""
    v = candidate.get("value_normalized")
    if v is None:
        v = candidate.get("value")
    if isinstance(v, (int, float)):
        return float(v), float(v)
    if isinstance(v, list) and len(v) >= 2:
        return float(v[0]), float(v[1])
    return None, None


class CandidateValidator:
    """Validate extraction candidates and classify them as passed/pending/rejected.

    Call ``validate(candidates, crop_info)`` where each candidate is a dict
    produced by the extraction pipeline (optionally after UnitNormalizer).

    Returns::

        {
            "passed":   [list of dicts],
            "pending":  [list of dicts, each with a "pending_reason" key],
            "rejected": [list of dicts, each with a "reject_reason" key],
        }
    """

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _has_source(c: dict) -> bool:
        s = c.get("source", "")
        return isinstance(s, str) and s.strip() != ""

    @staticmethod
    def _has_evidence(c: dict) -> bool:
        e = c.get("evidence", "")
        return isinstance(e, str) and e.strip() != ""

    @staticmethod
    def _unit_clear(c: dict) -> bool:
        uc = c.get("unit_conversion", {})
        if isinstance(uc, dict):
            return bool(uc.get("converted", False))
        # If no unit_conversion key at all, check unit_normalized
        un = c.get("unit_normalized", "")
        return bool(un and un != "unknown")

    @staticmethod
    def _confidence_ok(c: dict) -> bool:
        conf = c.get("confidence")
        if conf is None:
            return False
        return float(conf) >= _CONFIDENCE_PASS_THRESHOLD

    @staticmethod
    def _confidence_pending(c: dict) -> bool:
        conf = c.get("confidence")
        if conf is None:
            return False
        return _CONFIDENCE_PENDING_LOWER <= float(conf) < _CONFIDENCE_PASS_THRESHOLD

    @staticmethod
    def _physically_sensible(c: dict) -> tuple[bool, str]:
        """Check whether the value range is physically sensible.

        Returns (ok, reason_if_not_ok).
        """
        param = c.get("param", "")
        pkey = _param_key(param)
        if pkey is None:
            return True, ""  # unknown param → do not reject on physics

        lo, hi = _get_range(c)
        if lo is None or hi is None:
            return False, "value missing or unparseable"

        bounds = _PHYSICAL_BOUNDS[pkey]
        if lo < bounds[0] or hi > bounds[1]:
            return False, (
                f"{pkey} range [{lo}, {hi}] outside physical bounds "
                f"[{bounds[0]}, {bounds[1]}]"
            )

        if lo > hi:
            return False, f"range min ({lo}) > max ({hi})"

        # pH: optimal range width must be > 0.3
        if pkey == "ph" and (hi - lo) <= 0.3:
            return False, f"pH optimal range width {hi - lo:.2f} ≤ 0.3 (too narrow)"

        return True, ""

    # ------------------------------------------------------------------ #
    # Core classification logic                                            #
    # ------------------------------------------------------------------ #

    def _classify(self, c: dict) -> tuple[str, str]:
        """Return ('passed'|'pending'|'rejected', reason_str)."""

        context = c.get("context", "")

        # ---- Reject on missing critical metadata ----
        if not self._has_source(c):
            return "rejected", "missing_source"
        if not self._has_evidence(c):
            return "rejected", "missing_evidence"

        # ---- Check physical sanity ----
        phys_ok, phys_reason = self._physically_sensible(c)
        if not phys_ok:
            return "rejected", f"physical_range_invalid: {phys_reason}"

        # ---- Soil context → always pending ----
        if str(context).lower() == "soil":
            return "pending", "soil_context"

        # ---- ppm with unknown scale → pending ----
        uc = c.get("unit_conversion", {})
        if isinstance(uc, dict):
            rule = uc.get("rule", "")
            if "ppm" in str(rule).lower() and not uc.get("converted", True):
                return "pending", "ppm_unknown_scale"

        # ---- Unit not clear → pending (not hard reject) ----
        if not self._unit_clear(c):
            return "pending", "unit_not_normalized"

        # ---- Confidence checks ----
        if not self._confidence_ok(c):
            if self._confidence_pending(c):
                return "pending", "confidence_below_threshold"
            return "rejected", "confidence_too_low"

        # ---- General context (not explicitly hydroponic) → pending ----
        if str(context).lower() == "general":
            return "pending", "context_general_not_hydroponic"

        # ---- Water/air temperature proxy ----
        param = c.get("param", "").lower()
        if "water_temp" in param or "t_water" in param:
            available = c.get("available_as", "")
            if "air_temp" in str(available).lower():
                return "pending", "water_temp_from_air_temp_only"

        # ---- All checks passed ----
        return "passed", ""

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def validate(self, candidates: list[dict], crop_info: dict) -> dict:  # noqa: ARG002
        """Classify candidates and return a result dict.

        ``crop_info`` is accepted for future extensibility (e.g. crop-specific
        physical bounds overrides) but is not used in the current implementation.

        Returns::

            {
                "passed":   [...],
                "pending":  [...],   # each has "pending_reason"
                "rejected": [...],   # each has "reject_reason"
            }
        """
        passed: list[dict] = []
        pending: list[dict] = []
        rejected: list[dict] = []

        for cand in candidates:
            c = dict(cand)
            status, reason = self._classify(c)

            if status == "passed":
                passed.append(c)
            elif status == "pending":
                c["pending_reason"] = reason
                pending.append(c)
            else:
                c["reject_reason"] = reason
                rejected.append(c)

        return {"passed": passed, "pending": pending, "rejected": rejected}
