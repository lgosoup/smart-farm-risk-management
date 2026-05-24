"""Sentence splitting for Korean agricultural documents."""

import re


class SentenceSplitter:
    """Splits Korean text into individual sentences.

    Korean sentence boundaries are more varied than English ones.  This
    splitter handles the most common endings found in agricultural technical
    documents and adds newline-based splitting for document-structure sentences
    (bullet lists, table rows, etc.).
    """

    MIN_LENGTH: int = 8  # minimum character length to keep a sentence

    # Sentence-ending patterns common in Korean technical writing.
    # The pattern captures the ending marker so we can re-attach it to the
    # preceding sentence rather than losing it.
    # Order matters: longer / more specific patterns first.
    _ENDINGS = [
        r"다\.\s",      # 다.  (declarative verb ending — most common)
        r"임\.\s",      # 임.  (noun predicate ending)
        r"함\.\s",      # 함.  (nominalised verb ending)
        r"음\.\s",      # 음.  (another nominalised ending)
        r"됨\.\s",      # 됨.  (passive nominalised)
        r"함\.",        # 함.  at end of string / before newline
        r"임\.",        # 임.  at end of string
        r"됨\.",        # 됨.  at end of string
        r"다\.",        # 다.  at end of string
        r"。",          # Chinese/Japanese period
        r"\.\s",        # plain period + whitespace (English-style)
    ]

    # Combined pattern: split *after* any ending sequence.
    # We use a positive look-behind so the delimiter is consumed but kept.
    _SPLIT_RE: re.Pattern = re.compile(
        r"(?<=[다임함음됨]\.)\s+"   # Korean verb-noun endings before whitespace
        r"|(?<=。)"                  # Chinese ideographic period
        r"|(?<=\.\s)\s*",            # plain period + spaces
        re.UNICODE,
    )

    # Secondary pattern: split on bare periods that are sentence-final
    # (not decimal points).  A decimal point is preceded and followed by digits.
    _PERIOD_RE: re.Pattern = re.compile(
        r"(?<![0-9])\.(?![0-9])\s+",
        re.UNICODE,
    )

    def split(self, text: str) -> list[str]:
        """Split *text* into a list of sentences.

        Strategy:
        1. Split on newlines first (preserves document structure).
        2. Within each line, apply sentence-boundary regex.
        3. Filter out fragments shorter than MIN_LENGTH.
        4. Strip and deduplicate consecutive whitespace inside each sentence.
        """
        if not text:
            return []

        sentences: list[str] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            # Split the line on recognised Korean/Chinese endings
            parts = self._split_on_endings(line)
            for part in parts:
                part = part.strip()
                if part and len(part) >= self.MIN_LENGTH:
                    sentences.append(part)

        return sentences

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_on_endings(self, text: str) -> list[str]:
        """Apply sentence-boundary splitting within a single line of text."""
        # Use the combined look-behind pattern first
        parts = self._SPLIT_RE.split(text)

        # If still only one fragment, try the bare period pattern as a fallback
        if len(parts) == 1:
            parts = self._PERIOD_RE.split(text)

        # Re-attach any trailing punctuation that was split off as an empty
        # fragment (artefact of look-behind not consuming the delimiter).
        cleaned: list[str] = []
        for part in parts:
            part = part.strip()
            if part:
                cleaned.append(part)

        return cleaned if cleaned else [text]
