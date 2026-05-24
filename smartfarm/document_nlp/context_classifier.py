"""Context classification for extracted numerical values in Korean agricultural text."""

from __future__ import annotations

import re
from typing import Optional


# ---------------------------------------------------------------------------
# Keyword lookup tables
# ---------------------------------------------------------------------------

# --- Field identification ---

_PH_KEYWORDS = re.compile(r"pH|산도|수소이온농도|산성|알칼리", re.IGNORECASE)
_EC_KEYWORDS = re.compile(
    r"EC|전기전도도|전기\s*전도도|양액농도|양액\s*농도|배양액\s*농도|dS/m|mS/cm|µS/cm|μS/cm",
    re.IGNORECASE,
)
_T_WATER_KEYWORDS = re.compile(r"수온|배양액\s*온도|근권온도|양액\s*온도")
_T_AIR_KEYWORDS = re.compile(r"기온|주간온도|야간온도|생육온도|적정온도|대기\s*온도|실내\s*온도")
_DO_KEYWORDS = re.compile(r"DO|용존산소|dissolved\s+oxygen", re.IGNORECASE)
_TURBIDITY_KEYWORDS = re.compile(r"NTU|탁도|탁수")
_FLOW_RATE_KEYWORDS = re.compile(r"유량|유속|flow\s+rate", re.IGNORECASE)

# Generic temperature keyword (fallback — used when specific sub-types don't match)
_TEMP_GENERIC = re.compile(r"온도|℃|°C")

# --- Context identification ---

_HYDROPONIC_KEYWORDS = re.compile(
    r"수경재배|양액재배|배양액|nutrient\s+solution|수경|NFT|DFT|담액|양액|영양액",
    re.IGNORECASE,
)
_SOIL_KEYWORDS = re.compile(r"노지|토경|흙|토양|밭|포장|관비", re.IGNORECASE)

# --- Growth stage identification ---

_SEEDLING_KEYWORDS = re.compile(r"유묘|파종|발아|육묘|모종|어린\s*묘")
_VEGETATIVE_KEYWORDS = re.compile(r"정식\s*후|생육기|생육\s*중기|영양\s*생장|엽면|성장기|생육\s*전기")
_FRUITING_KEYWORDS = re.compile(r"착과|수확|개화|결실|생식\s*생장|과실|출수")

# --- Unit-to-field hints ---

_UNIT_FIELD_MAP: dict[str, str] = {
    "pH": "pH",
    "dS/m": "EC",
    "mS/cm": "EC",
    "µS/cm": "EC",
    "mg/L": "DO",   # primary association; overridden by keyword context
    "ppm": "EC",    # often used for nutrient-solution concentration
    "NTU": "Turbidity",
    "°C": "T_air",  # default temperature type; refined by keyword context
    "%": "unknown",
}


class ContextClassifier:
    """Classify extracted value dicts with field, growth stage, context, and
    confidence information.

    The classifier uses a rule-based approach that scores each candidate value
    against keyword evidence found in the surrounding sentence.  Confidence
    is accumulated as a float in ``[0.0, 1.0]``.
    """

    # Maximum raw confidence points before capping at 1.0
    _MAX_POINTS: float = 4.0

    def classify(
        self, sentence: str, extracted_values: list[dict]
    ) -> list[dict]:
        """Enrich *extracted_values* with classification metadata.

        Parameters
        ----------
        sentence:
            The original sentence from which the values were extracted.
        extracted_values:
            List of dicts as returned by ``ValueExtractor.extract``.

        Returns
        -------
        The same list (new dicts, not mutated in-place) with four additional
        keys added to each element:

        * ``field``        — one of "pH", "EC", "T_air", "T_water",
                             "T_root", "DO", "FR", "Turbidity", "unknown"
        * ``growth_stage`` — "seedling", "vegetative", "fruiting", "all"
        * ``context``      — "hydroponic", "soil", "general", "unknown"
        * ``confidence``   — float in [0.0, 1.0]
        """
        growth_stage = self._detect_stage(sentence)
        context = self._detect_context(sentence)

        result: list[dict] = []
        for val in extracted_values:
            classified = dict(val)  # shallow copy — don't mutate caller's data
            field, confidence = self._classify_field(sentence, val)
            classified["field"] = field
            classified["growth_stage"] = growth_stage
            classified["context"] = context
            classified["confidence"] = round(min(confidence, 1.0), 3)
            result.append(classified)

        return result

    # ------------------------------------------------------------------
    # Field classification
    # ------------------------------------------------------------------

    def _classify_field(
        self, sentence: str, val: dict
    ) -> tuple[str, float]:
        """Return (field_name, confidence) for a single extracted value."""
        unit: Optional[str] = val.get("unit")
        numeric = val.get("value") or val.get("min")
        points: float = 0.0

        # Pre-compute keyword matches (used across multiple branches)
        ph_match = bool(_PH_KEYWORDS.search(sentence))
        ec_match = bool(_EC_KEYWORDS.search(sentence))
        do_match = bool(_DO_KEYWORDS.search(sentence))
        temp_water_match = bool(_T_WATER_KEYWORDS.search(sentence))
        temp_air_match = bool(_T_AIR_KEYWORDS.search(sentence))
        temp_generic_match = bool(_TEMP_GENERIC.search(sentence))
        turbidity_match = bool(_TURBIDITY_KEYWORDS.search(sentence))
        flow_match = bool(_FLOW_RATE_KEYWORDS.search(sentence))

        ec_unit = unit in ("dS/m", "mS/cm", "µS/cm", "ppm")
        do_unit = unit == "mg/L"
        temp_unit = unit == "°C"
        turbidity_unit = unit == "NTU"

        # ------------------------------------------------------------------
        # Phase 1: explicit unit wins regardless of keyword context in
        # multi-field sentences (e.g. a sentence that mentions both pH and EC).
        # ------------------------------------------------------------------

        if unit == "pH":
            # Explicit pH unit — highest confidence
            points = 2.0
            if ph_match:
                points += 1.5
            return "pH", min(points / self._MAX_POINTS, 1.0)

        if ec_unit:
            points = 2.0
            if ec_match:
                points += 1.5
            return "EC", min(points / self._MAX_POINTS, 1.0)

        if turbidity_unit:
            points = 2.0
            if turbidity_match:
                points += 1.5
            return "Turbidity", min(points / self._MAX_POINTS, 1.0)

        if temp_unit:
            # Disambiguate temperature sub-type using keyword context
            base_points = 2.0
            if temp_water_match:
                if "근권" in sentence:
                    return "T_root", min((base_points + 1.5) / self._MAX_POINTS, 1.0)
                return "T_water", min((base_points + 1.5) / self._MAX_POINTS, 1.0)
            if temp_air_match:
                return "T_air", min((base_points + 1.5) / self._MAX_POINTS, 1.0)
            # Generic temperature → default air
            return "T_air", min((base_points + 0.5) / self._MAX_POINTS, 1.0)

        if do_unit and do_match:
            # mg/L + DO keyword — very high confidence
            return "DO", min(3.5 / self._MAX_POINTS, 1.0)

        # ------------------------------------------------------------------
        # Phase 2: keyword-only classification (no explicit unit)
        # ------------------------------------------------------------------

        # --- pH (keyword match + plausible numeric range) ---
        ph_range_ok = (
            numeric is not None and 0.0 <= numeric <= 14.0
        ) or (
            val.get("min") is not None and val.get("max") is not None
            and 0.0 <= val["min"] <= 14.0 and 0.0 <= val["max"] <= 14.0
        )
        if ph_match and not ec_match:
            # pH keyword but no EC keyword → confidently pH
            points = 1.5 + (0.5 if ph_range_ok else 0.0)
            return "pH", min(points / self._MAX_POINTS, 1.0)

        if ph_match and ec_match:
            # Both keywords present — use numeric range to disambiguate.
            # EC values are typically > 0.5 and often > 1.0 in agricultural use,
            # while pH is always [0, 14].  Without a unit, fall back to pH only
            # when the value is clearly in the pH range AND not plausibly EC.
            if ph_range_ok and (numeric is None or numeric <= 14.0):
                # Ambiguous — lower confidence, label as pH (most common fieldless case)
                return "pH", min(1.0 / self._MAX_POINTS, 1.0)

        # --- EC keyword only ---
        if ec_match:
            points = 1.5
            return "EC", min(points / self._MAX_POINTS, 1.0)

        # --- DO keyword (unit already excluded above) ---
        if do_match or do_unit:
            points = (2.0 if do_unit else 0.0) + (1.5 if do_match else 0.0)
            return "DO", min(points / self._MAX_POINTS, 1.0)

        # --- Turbidity keyword ---
        if turbidity_match:
            return "Turbidity", min(1.5 / self._MAX_POINTS, 1.0)

        # --- Flow rate ---
        if flow_match:
            return "FR", min(1.5 / self._MAX_POINTS, 1.0)

        # --- Temperature keyword (no unit) ---
        if temp_generic_match:
            base_points = 0.5
            if temp_water_match:
                if "근권" in sentence:
                    return "T_root", min((base_points + 1.5) / self._MAX_POINTS, 1.0)
                return "T_water", min((base_points + 1.5) / self._MAX_POINTS, 1.0)
            if temp_air_match:
                return "T_air", min((base_points + 1.5) / self._MAX_POINTS, 1.0)
            return "T_air", min(base_points / self._MAX_POINTS, 1.0)

        # --- Fallback: use unit hint map ---
        if unit and unit in _UNIT_FIELD_MAP:
            return _UNIT_FIELD_MAP[unit], 0.3

        return "unknown", 0.0

    # ------------------------------------------------------------------
    # Growth stage detection
    # ------------------------------------------------------------------

    def _detect_stage(self, sentence: str) -> str:
        """Detect the growth stage referenced in *sentence*."""
        if _SEEDLING_KEYWORDS.search(sentence):
            return "seedling"
        if _VEGETATIVE_KEYWORDS.search(sentence):
            return "vegetative"
        if _FRUITING_KEYWORDS.search(sentence):
            return "fruiting"
        return "all"

    # ------------------------------------------------------------------
    # Context detection
    # ------------------------------------------------------------------

    def _detect_context(self, sentence: str) -> str:
        """Detect the cultivation context of *sentence*."""
        hydro = bool(_HYDROPONIC_KEYWORDS.search(sentence))
        soil = bool(_SOIL_KEYWORDS.search(sentence))

        if hydro and not soil:
            return "hydroponic"
        if soil and not hydro:
            return "soil"
        if hydro and soil:
            # Both mentioned — general statement
            return "general"
        return "unknown"
