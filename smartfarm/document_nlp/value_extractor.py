"""Numerical value extraction from Korean agricultural sentences."""

from __future__ import annotations

import re
from typing import Optional


# ---------------------------------------------------------------------------
# Unit definitions
# ---------------------------------------------------------------------------

# Units ordered longest → shortest so the regex engine is greedy.
_UNITS: list[str] = [
    "µS/cm", "μS/cm",  # micro-siemens (two Unicode micro-sign variants)
    "mS/cm",
    "dS/m",
    "mg/L",
    "mg/l",
    "ppm",
    "NTU",
    "pH",
    "°C",
    "℃",
    "%",
]

# Canonical mapping from raw unit string to a normalised key used downstream.
UNIT_CANONICAL: dict[str, str] = {
    "µS/cm": "µS/cm",
    "μS/cm": "µS/cm",
    "mS/cm": "mS/cm",
    "dS/m": "dS/m",
    "mg/L": "mg/L",
    "mg/l": "mg/L",
    "ppm": "ppm",
    "NTU": "NTU",
    "pH": "pH",
    "°C": "°C",
    "℃": "°C",
    "%": "%",
}

# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------

# Number fragment: integer or decimal.
_NUM = r"\d+(?:\.\d+)?"

# Range separator: en-dash, em-dash, tilde, or hyphen, surrounded by
# optional whitespace.
_SEP = r"\s*[-–—~]\s*"

# Unit pattern: any recognised unit (longest-first, optional surrounding space)
_UNIT_PAT = "(?:" + "|".join(re.escape(u) for u in _UNITS) + ")"

# Full value pattern: optional range, optional unit.
# Group layout (named):
#   lo   — low value of a range (or sole value when no separator)
#   sep  — the separator (present only for ranges)
#   hi   — high value of a range
#   unit — optional unit string
_VALUE_RE = re.compile(
    rf"(?P<lo>{_NUM})"
    rf"(?:(?P<sep>{_SEP})(?P<hi>{_NUM}))?"
    rf"(?:\s*(?P<unit>{_UNIT_PAT}))?",
    re.IGNORECASE | re.UNICODE,
)

# Standalone unit look-ahead: detect a unit that appears *before* any number
# (e.g. "pH 5.5") so we can associate it correctly.
_LEADING_UNIT_RE = re.compile(
    rf"(?P<unit>{_UNIT_PAT})\s+(?P<lo>{_NUM})"
    rf"(?:(?P<sep>{_SEP})(?P<hi>{_NUM}))?",
    re.IGNORECASE | re.UNICODE,
)


class ValueExtractor:
    """Extract numerical measurements and their units from a sentence.

    Two pass strategy
    -----------------
    1. Find all occurrences of the pattern ``unit number`` (leading-unit form,
       e.g. "pH 5.5–6.5") and record them.
    2. Find all occurrences of the pattern ``number [unit]`` (trailing-unit
       form, e.g. "5.5–6.5 pH") and record them.

    Matches from both passes are merged; overlapping spans from the second
    pass are skipped if they were already covered by the first pass.
    """

    def extract(self, sentence: str) -> list[dict]:
        """Return a list of value records found in *sentence*.

        Each record has the shape::

            {
              "raw":      str,           # the matched text
              "min":      float | None,  # low bound of a range
              "max":      float | None,  # high bound of a range
              "value":    float | None,  # single value (None when range)
              "unit":     str | None,    # detected unit (canonical form)
              "position": int,           # character offset in *sentence*
            }
        """
        if not sentence:
            return []

        results: list[dict] = []
        covered_spans: list[tuple[int, int]] = []

        # --- Pass 1: leading-unit form (pH 5.5–6.5) ---
        for m in _LEADING_UNIT_RE.finditer(sentence):
            span = m.span()
            record = self._build_record(m, span[0])
            if record is not None:
                results.append(record)
                covered_spans.append(span)

        # --- Pass 2: trailing-unit form (5.5–6.5 pH) ---
        for m in _VALUE_RE.finditer(sentence):
            span = m.span()
            if self._overlaps(span, covered_spans):
                continue
            record = self._build_record(m, span[0])
            if record is not None:
                results.append(record)
                covered_spans.append(span)

        # Sort by position in text
        results.sort(key=lambda r: r["position"])
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_record(
        self, m: re.Match, position: int
    ) -> Optional[dict]:
        """Convert a regex match to a value record dict."""
        lo_str = m.group("lo")
        hi_str = m.group("hi") if "hi" in m.groupdict() else None
        sep_str = m.group("sep") if "sep" in m.groupdict() else None
        raw_unit = m.group("unit") if "unit" in m.groupdict() else None

        # Skip bare integers/floats with no unit and no range separator when
        # the matched number is just a single digit (too ambiguous).
        if lo_str is None:
            return None

        try:
            lo = float(lo_str)
        except ValueError:
            return None

        is_range = (sep_str is not None) and (hi_str is not None)

        hi: Optional[float] = None
        if is_range:
            try:
                hi = float(hi_str)
            except ValueError:
                is_range = False

        # Normalise unit
        canonical_unit: Optional[str] = None
        if raw_unit:
            canonical_unit = UNIT_CANONICAL.get(raw_unit, raw_unit)

        # Build raw text snippet
        raw = m.group(0).strip()

        record: dict = {
            "raw": raw,
            "min": lo if is_range else None,
            "max": hi if is_range else None,
            "value": None if is_range else lo,
            "unit": canonical_unit,
            "position": position,
        }
        return record

    @staticmethod
    def _overlaps(
        span: tuple[int, int], covered: list[tuple[int, int]]
    ) -> bool:
        """Return True if *span* overlaps any interval in *covered*."""
        s, e = span
        for cs, ce in covered:
            # Overlap when intervals share at least one character position
            if s < ce and e > cs:
                return True
        return False
