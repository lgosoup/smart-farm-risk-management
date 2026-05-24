"""Keyword-based sentence filtering for Korean agricultural documents."""

from __future__ import annotations

import re
from typing import Optional


# ---------------------------------------------------------------------------
# Keyword groups
# ---------------------------------------------------------------------------

KEYWORD_GROUPS: dict[str, list[str]] = {
    "pH": ["pH", "산도", "수소이온농도", "산성", "알칼리", "ph"],
    "EC": [
        "EC",
        "전기전도도",
        "양액농도",
        "배양액 농도",
        "양액 농도",
        "dS/m",
        "mS/cm",
        "전기 전도도",
    ],
    "temperature": [
        "온도",
        "기온",
        "수온",
        "근권온도",
        "배양액 온도",
        "주간온도",
        "야간온도",
        "℃",
        "°C",
        "생육온도",
        "적정온도",
    ],
    "DO": ["DO", "용존산소", "산소", "dissolved oxygen"],
    "cultivation": [
        "수경재배",
        "양액재배",
        "배양액",
        "nutrient solution",
        "수경",
        "NFT",
        "DFT",
        "담액",
    ],
    "disease_risk": [
        "고온",
        "저온",
        "과습",
        "뿌리썩음",
        "시들음병",
        "추대",
        "생리장해",
        "병해",
        "고온 장해",
        "저온 장해",
    ],
    "EC_context": ["양액", "배양액", "영양액", "배지"],
}

_TOTAL_GROUPS: int = len(KEYWORD_GROUPS)


class KeywordFilter:
    """Filter sentences that contain agronomically relevant keywords.

    Each keyword group is pre-compiled into a single OR-regex for speed.
    The ``filter`` method returns only sentences that match at least one
    target group, together with the list of matched group names and a
    normalised relevance score.
    """

    def __init__(self) -> None:
        # Pre-compile one pattern per keyword group.
        self._patterns: dict[str, re.Pattern] = {}
        for group_name, keywords in KEYWORD_GROUPS.items():
            # Sort longest keyword first so the regex engine matches the most
            # specific alternative before a shorter prefix.
            sorted_kws = sorted(keywords, key=len, reverse=True)
            pattern = "|".join(re.escape(kw) for kw in sorted_kws)
            self._patterns[group_name] = re.compile(pattern, re.IGNORECASE)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter(
        self,
        sentences: list[str],
        target_fields: Optional[list[str]] = None,
    ) -> list[dict]:
        """Return sentences that match one or more keyword groups.

        Parameters
        ----------
        sentences:
            Plain-text sentences (output of ``SentenceSplitter.split``).
        target_fields:
            Optional whitelist of group names (keys in ``KEYWORD_GROUPS``).
            When provided, only the listed groups are considered.  Sentences
            that match *other* groups are still returned if they also match a
            target group.

        Returns
        -------
        list of dicts, each with:
            ``sentence``       — the original sentence string
            ``matched_fields`` — list of group names that matched
            ``score``          — float in [0, 1], unique groups matched /
                                 total groups defined
        """
        active_patterns = self._select_patterns(target_fields)
        results: list[dict] = []

        for sentence in sentences:
            matched = self._match_sentence(sentence, active_patterns)
            if matched:
                score = len(matched) / _TOTAL_GROUPS
                results.append(
                    {
                        "sentence": sentence,
                        "matched_fields": matched,
                        "score": round(score, 4),
                    }
                )

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_patterns(
        self, target_fields: Optional[list[str]]
    ) -> dict[str, re.Pattern]:
        """Return the subset of compiled patterns to use."""
        if target_fields is None:
            return self._patterns
        return {
            name: pat
            for name, pat in self._patterns.items()
            if name in target_fields
        }

    def _match_sentence(
        self, sentence: str, patterns: dict[str, re.Pattern]
    ) -> list[str]:
        """Return the sorted list of group names that match *sentence*."""
        matched: list[str] = []
        for group_name, pattern in patterns.items():
            if pattern.search(sentence):
                matched.append(group_name)
        return matched
