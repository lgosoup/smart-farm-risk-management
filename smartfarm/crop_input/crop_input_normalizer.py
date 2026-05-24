"""Crop input normalization module.

Accepts crop name strings in Korean or English and returns a normalized
profile dict suitable for downstream fuzzy-logic and RAG pipelines.
"""

from __future__ import annotations

import unicodedata
from typing import Optional


# ---------------------------------------------------------------------------
# Internal lookup table
# ---------------------------------------------------------------------------
# Each entry contains all surface-form aliases that map to the canonical
# record.  Keys are lowercased for case-insensitive matching.

_CROP_DB: list[dict] = [
    {
        "aliases_ko": ["바질"],
        "aliases_en": ["basil", "genovese basil", "sweet basil", "italian basil"],
        "crop_common_name_ko": "바질",
        "crop_common_name_en": "basil",
        "scientific_name": "Ocimum basilicum",
        "cultivar": None,
        "growth_type": "leafy herb",
        "edible_part": "leaf",
        "cultivation_mode": "hydro",
    },
    {
        "aliases_ko": ["상추", "로메인상추", "로메인"],
        "aliases_en": ["lettuce", "romaine lettuce", "romaine", "butterhead lettuce",
                       "leaf lettuce", "iceberg lettuce"],
        "crop_common_name_ko": "상추",
        "crop_common_name_en": "lettuce",
        "scientific_name": "Lactuca sativa",
        "cultivar": None,
        "growth_type": "leafy green",
        "edible_part": "leaf",
        "cultivation_mode": "hydro",
    },
    {
        "aliases_ko": ["딸기"],
        "aliases_en": ["strawberry", "garden strawberry"],
        "crop_common_name_ko": "딸기",
        "crop_common_name_en": "strawberry",
        "scientific_name": "Fragaria × ananassa",
        "cultivar": None,
        "growth_type": "fruiting",
        "edible_part": "fruit",
        "cultivation_mode": "hydro",
    },
    {
        "aliases_ko": ["토마토", "방울토마토"],
        "aliases_en": ["tomato", "cherry tomato", "plum tomato", "beefsteak tomato",
                       "grape tomato"],
        "crop_common_name_ko": "토마토",
        "crop_common_name_en": "tomato",
        "scientific_name": "Solanum lycopersicum",
        "cultivar": None,
        "growth_type": "fruiting",
        "edible_part": "fruit",
        "cultivation_mode": "hydro",
    },
    {
        "aliases_ko": ["와사비", "고추냉이"],
        "aliases_en": ["wasabi", "japanese horseradish"],
        "crop_common_name_ko": "와사비",
        "crop_common_name_en": "wasabi",
        "scientific_name": "Eutrema japonicum",
        "cultivar": None,
        "growth_type": "root herb",
        "edible_part": "rhizome",
        "cultivation_mode": "hydro",
    },
    {
        "aliases_ko": ["파프리카", "피망"],
        "aliases_en": ["paprika", "bell pepper", "sweet pepper", "capsicum"],
        "crop_common_name_ko": "파프리카",
        "crop_common_name_en": "paprika",
        "scientific_name": "Capsicum annuum",
        "cultivar": None,
        "growth_type": "fruiting",
        "edible_part": "fruit",
        "cultivation_mode": "hydro",
    },
    {
        "aliases_ko": ["오이"],
        "aliases_en": ["cucumber", "english cucumber", "persian cucumber",
                       "pickling cucumber"],
        "crop_common_name_ko": "오이",
        "crop_common_name_en": "cucumber",
        "scientific_name": "Cucumis sativus",
        "cultivar": None,
        "growth_type": "fruiting vine",
        "edible_part": "fruit",
        "cultivation_mode": "hydro",
    },
    {
        "aliases_ko": ["시금치"],
        "aliases_en": ["spinach", "baby spinach"],
        "crop_common_name_ko": "시금치",
        "crop_common_name_en": "spinach",
        "scientific_name": "Spinacia oleracea",
        "cultivar": None,
        "growth_type": "leafy green",
        "edible_part": "leaf",
        "cultivation_mode": "hydro",
    },
    {
        "aliases_ko": ["민트", "박하", "페퍼민트", "스피어민트"],
        "aliases_en": ["mint", "peppermint", "spearmint", "garden mint"],
        "crop_common_name_ko": "민트",
        "crop_common_name_en": "mint",
        "scientific_name": "Mentha spp.",
        "cultivar": None,
        "growth_type": "leafy herb",
        "edible_part": "leaf",
        "cultivation_mode": "hydro",
    },
    {
        "aliases_ko": ["케일"],
        "aliases_en": ["kale", "curly kale", "lacinato kale", "dinosaur kale",
                       "tuscan kale", "cavolo nero"],
        "crop_common_name_ko": "케일",
        "crop_common_name_en": "kale",
        "scientific_name": "Brassica oleracea var. sabellica",
        "cultivar": None,
        "growth_type": "leafy green",
        "edible_part": "leaf",
        "cultivation_mode": "hydro",
    },
]

# Build flat alias → record index maps at module load time for O(1) lookups.
_KO_ALIAS_MAP: dict[str, int] = {}
_EN_ALIAS_MAP: dict[str, int] = {}

for _idx, _entry in enumerate(_CROP_DB):
    for _alias in _entry["aliases_ko"]:
        _KO_ALIAS_MAP[_alias.strip()] = _idx
    for _alias in _entry["aliases_en"]:
        _EN_ALIAS_MAP[_alias.strip().lower()] = _idx


# ---------------------------------------------------------------------------
# Language detection helpers
# ---------------------------------------------------------------------------

def _is_korean(text: str) -> bool:
    """Return True if *text* contains at least one Hangul character."""
    for ch in text:
        name = unicodedata.name(ch, "")
        if "HANGUL" in name:
            return True
    return False


def _looks_latin(text: str) -> bool:
    """Return True if *text* is predominantly ASCII/Latin characters."""
    latin_count = sum(1 for ch in text if ord(ch) < 256 and ch.isalpha())
    return latin_count >= len(text.replace(" ", "")) * 0.5


# ---------------------------------------------------------------------------
# Normalizer class
# ---------------------------------------------------------------------------

class CropInputNormalizer:
    """Normalize a crop name string (Korean or English) into a standard profile.

    Usage::

        normalizer = CropInputNormalizer()
        profile = normalizer.normalize("바질")
        profile = normalizer.normalize("Genovese basil")
    """

    def normalize(self, crop_name: str) -> dict:
        """Normalize *crop_name* and return a crop profile dict.

        Parameters
        ----------
        crop_name:
            Crop name in Korean or English (case-insensitive).

        Returns
        -------
        dict
            Keys: ``crop_common_name_ko``, ``crop_common_name_en``,
            ``scientific_name``, ``cultivar``, ``growth_type``,
            ``edible_part``, ``cultivation_mode``, ``confidence``.
            An additional ``low_confidence`` flag is set to ``True`` when the
            crop was not found in the lookup table.
        """
        if not crop_name or not isinstance(crop_name, str):
            raise ValueError(f"crop_name must be a non-empty string, got: {crop_name!r}")

        stripped = crop_name.strip()
        record_idx = self._lookup(stripped)

        if record_idx is not None:
            return self._build_result(_CROP_DB[record_idx], confidence=1.0)

        # Attempt partial / fuzzy lookup
        record_idx = self._partial_lookup(stripped)
        if record_idx is not None:
            return self._build_result(_CROP_DB[record_idx], confidence=0.7,
                                      low_confidence=True)

        # Unknown crop — return a best-guess skeleton
        return self._build_unknown(stripped)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _lookup(self, name: str) -> Optional[int]:
        """Exact alias lookup (case-insensitive, whitespace-normalised)."""
        # Korean path
        if _is_korean(name):
            return _KO_ALIAS_MAP.get(name)

        # English / Latin path
        lower = name.lower()
        idx = _EN_ALIAS_MAP.get(lower)
        if idx is not None:
            return idx

        # Also try stripping common adjective prefixes such as "fresh ", "organic "
        for prefix in ("fresh ", "organic ", "baby ", "dried ", "frozen "):
            if lower.startswith(prefix):
                idx = _EN_ALIAS_MAP.get(lower[len(prefix):])
                if idx is not None:
                    return idx

        return None

    def _partial_lookup(self, name: str) -> Optional[int]:
        """Return the first record whose any alias is a substring of *name*
        or vice-versa."""
        lower = name.lower()

        if _is_korean(name):
            for alias, idx in _KO_ALIAS_MAP.items():
                if alias in name or name in alias:
                    return idx
        else:
            for alias, idx in _EN_ALIAS_MAP.items():
                if alias in lower or lower in alias:
                    return idx

        return None

    @staticmethod
    def _build_result(record: dict, *, confidence: float,
                      low_confidence: bool = False) -> dict:
        """Build the output dict from a DB record."""
        result: dict = {
            "crop_common_name_ko": record["crop_common_name_ko"],
            "crop_common_name_en": record["crop_common_name_en"],
            "scientific_name": record["scientific_name"],
            "cultivar": record["cultivar"],
            "growth_type": record["growth_type"],
            "edible_part": record["edible_part"],
            "cultivation_mode": record["cultivation_mode"],
            "confidence": round(confidence, 4),
        }
        if low_confidence:
            result["low_confidence"] = True
        return result

    @staticmethod
    def _build_unknown(name: str) -> dict:
        """Return a skeleton profile for an unrecognised crop name."""
        is_ko = _is_korean(name)
        return {
            "crop_common_name_ko": name if is_ko else None,
            "crop_common_name_en": None if is_ko else name,
            "scientific_name": None,
            "cultivar": None,
            "growth_type": "unknown",
            "edible_part": "unknown",
            "cultivation_mode": "hydro",  # default for smart-farm context
            "confidence": 0.0,
            "low_confidence": True,
        }


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

_default_normalizer = CropInputNormalizer()


def normalize(crop_name: str) -> dict:
    """Normalize *crop_name* using a shared :class:`CropInputNormalizer` instance.

    Parameters
    ----------
    crop_name:
        Crop name in Korean or English.

    Returns
    -------
    dict
        Normalized crop profile. See :meth:`CropInputNormalizer.normalize`.
    """
    return _default_normalizer.normalize(crop_name)
