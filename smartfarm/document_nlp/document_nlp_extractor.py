"""Orchestration layer: full NLP pipeline for Korean agricultural documents."""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

from .text_cleaner import TextCleaner
from .sentence_splitter import SentenceSplitter
from .keyword_filter import KeywordFilter
from .value_extractor import ValueExtractor
from .context_classifier import ContextClassifier


class DocumentNLPExtractor:
    """Orchestrate the full NLP pipeline for one or more agricultural documents.

    Pipeline
    --------
    raw_text
        → ``TextCleaner.clean``          (remove tags, normalise whitespace)
        → ``SentenceSplitter.split``     (segment into sentences)
        → ``KeywordFilter.filter``       (retain agronomically relevant sentences)
        → ``ValueExtractor.extract``     (find numeric values + units)
        → ``ContextClassifier.classify`` (assign field, stage, context, confidence)
        → candidate records
    """

    def __init__(self) -> None:
        self.cleaner = TextCleaner()
        self.splitter = SentenceSplitter()
        self.filter = KeywordFilter()
        self.extractor = ValueExtractor()
        self.classifier = ContextClassifier()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, document: dict, crop_info: dict) -> list[dict]:
        """Process a single document and return extraction candidates.

        Parameters
        ----------
        document:
            ``{"title": str, "raw_text": str, "source": str, "source_type": str}``
        crop_info:
            ``{"crop_common_name_ko": str, "cultivation_mode": str, ...}``
            Currently used for context enrichment (future: crop-specific rules).

        Returns
        -------
        List of candidate dicts, each with the keys:

        * ``field``        — "pH", "EC", "T_air", "T_water", "T_root",
                             "DO", "FR", "Turbidity", or "unknown"
        * ``min``          — float or None (lower bound of a range)
        * ``max``          — float or None (upper bound of a range)
        * ``unit``         — canonical unit string or None
        * ``growth_stage`` — "seedling", "vegetative", "fruiting", "all"
        * ``evidence``     — the original sentence the value was drawn from
        * ``source``       — document source identifier
        * ``source_type``  — document source type (e.g. "nongsaro_api")
        * ``confidence``   — float in [0.0, 1.0]
        * ``context``      — "hydroponic", "soil", "general", or "unknown"
        """
        raw_text: str = document.get("raw_text", "")
        source: str = document.get("source", "")
        source_type: str = document.get("source_type", "")

        if not raw_text:
            return []

        # 1. Clean
        cleaned = self.cleaner.clean(raw_text)

        # 2. Split into sentences
        sentences = self.splitter.split(cleaned)

        # 3. Keyword filter (no field restriction — keep all relevant sentences)
        filtered = self.filter.filter(sentences, target_fields=None)

        candidates: list[dict] = []

        for item in filtered:
            sentence: str = item["sentence"]

            # 4. Extract numeric values
            values = self.extractor.extract(sentence)
            if not values:
                continue

            # 5. Classify each value
            classified_values = self.classifier.classify(sentence, values)

            for val in classified_values:
                # Skip values that are completely unidentified and have no unit
                if val["field"] == "unknown" and val["unit"] is None:
                    continue

                candidate: dict = {
                    "field": val["field"],
                    "min": val.get("min"),
                    "max": val.get("max"),
                    "unit": val.get("unit"),
                    "growth_stage": val["growth_stage"],
                    "evidence": sentence,
                    "source": source,
                    "source_type": source_type,
                    "confidence": val["confidence"],
                    "context": val["context"],
                }
                candidates.append(candidate)

        return candidates

    def extract_from_documents(
        self, doc_bundle: dict, crop_info: dict
    ) -> list[dict]:
        """Process multiple documents from a ``NongsaroCropTechApiClient`` bundle.

        Parameters
        ----------
        doc_bundle:
            ``{"documents": [<document>, ...]}``
        crop_info:
            Passed through to ``extract`` for each document.

        Returns
        -------
        De-duplicated list of candidates (highest confidence kept per
        ``(field, growth_stage)`` bucket).
        """
        all_candidates: list[dict] = []
        for doc in doc_bundle.get("documents", []):
            candidates = self.extract(doc, crop_info)
            all_candidates.extend(candidates)
        return self._deduplicate(all_candidates)

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def _deduplicate(self, candidates: list[dict]) -> list[dict]:
        """Remove exact duplicates; keep highest-confidence record per
        ``(field, growth_stage)`` bucket *within* the same numeric range.

        Two records are considered *exact duplicates* when they share the same
        ``field``, ``min``, ``max``, ``unit``, and ``evidence`` strings.
        Among non-exact-duplicate records that share ``(field, growth_stage)``
        *and* the same numeric range, the one with the highest confidence is
        returned.
        """
        # --- Step 1: Remove exact duplicates by a composite key ---
        seen_exact: set[tuple] = set()
        unique: list[dict] = []
        for c in candidates:
            key = (
                c.get("field"),
                c.get("min"),
                c.get("max"),
                c.get("unit"),
                c.get("evidence"),
                c.get("source"),
            )
            if key not in seen_exact:
                seen_exact.add(key)
                unique.append(c)

        # --- Step 2: Among records sharing (field, stage, min, max),
        #             keep highest-confidence ---
        # bucket key: (field, growth_stage, min, max, unit)
        bucket: dict[tuple, list[dict]] = defaultdict(list)
        for c in unique:
            bkey = (
                c.get("field"),
                c.get("growth_stage"),
                c.get("min"),
                c.get("max"),
                c.get("unit"),
            )
            bucket[bkey].append(c)

        deduplicated: list[dict] = []
        for records in bucket.values():
            best = max(records, key=lambda r: r.get("confidence", 0.0))
            deduplicated.append(best)

        # Restore a stable order: sort by field then confidence desc
        deduplicated.sort(
            key=lambda r: (r.get("field", ""), -r.get("confidence", 0.0))
        )
        return deduplicated
