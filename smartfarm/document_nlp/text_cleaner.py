"""Text cleaning utilities for Korean agricultural documents."""

import re


class TextCleaner:
    """Cleans raw Korean agricultural document text for downstream NLP processing."""

    # Mapping of Korean/Chinese punctuation variants to standard ASCII equivalents
    _PUNCT_MAP = {
        "，": ",",   # fullwidth comma ，
        "．": ".",   # fullwidth period ．
        "！": "!",   # fullwidth exclamation ！
        "？": "?",   # fullwidth question ？
        "：": ":",   # fullwidth colon ：
        "；": ";",   # fullwidth semicolon ；
        "「": "(",   # left corner bracket 「
        "」": ")",   # right corner bracket 」
        "『": "(",   # left white corner bracket 『
        "』": ")",   # right white corner bracket 』
        "【": "[",   # left black lenticular bracket 【
        "】": "]",   # right black lenticular bracket 】
        "‘": "'",   # left single quotation mark '
        "’": "'",   # right single quotation mark '
        "“": '"',   # left double quotation mark "
        "”": '"',   # right double quotation mark "
        "、": ",",   # ideographic comma 、
        "。": ".",   # ideographic full stop 。 — keep as period
        "（": "(",   # fullwidth left parenthesis （
        "）": ")",   # fullwidth right parenthesis ）
        "・": ".",   # katakana middle dot ・
    }

    # Compiled patterns (class-level for performance)
    _TAG_RE = re.compile(r"<[^>]+>")
    _MULTI_SPACE_RE = re.compile(r"[ \t]+")
    _MULTI_BLANK_LINE_RE = re.compile(r"\n{3,}")
    # Characters to keep: Korean (Hangul syllables + jamo), CJK, alphanumeric,
    # whitespace, and standard punctuation used in agricultural documents
    _ALLOWED_RE = re.compile(
        r"[^가-힣ᄀ-ᇿ㄰-㆏"  # Hangul syllables + jamo + compatibility jamo
        r"一-鿿"                               # CJK unified ideographs
        r"a-zA-Z0-9"
        r"\s"
        r".,!?;:()\[\]{}\'\"\-\–\—\~\/"
        r"℃°%±×÷≤≥<>="
        r"\+\*#@&"
        r"µμ"                                          # micro sign variants
        r"]"
    )

    def clean(self, text: str) -> str:
        """Return a cleaned version of *text*.

        Steps applied in order:
        1. Remove XML/HTML tags.
        2. Normalise Korean/Chinese punctuation variants to ASCII equivalents.
        3. Strip disallowed special characters.
        4. Collapse multiple spaces/tabs to a single space.
        5. Collapse runs of more than two consecutive blank lines to two.
        6. Strip leading/trailing whitespace from each line.
        """
        if not text:
            return ""

        # 1. Remove XML/HTML tags
        text = self._TAG_RE.sub(" ", text)

        # 2. Normalise Korean/Chinese punctuation
        for src, dst in self._PUNCT_MAP.items():
            text = text.replace(src, dst)

        # 3. Remove disallowed special characters
        text = self._ALLOWED_RE.sub("", text)

        # 4. Collapse multiple spaces/tabs (preserve newlines)
        lines = text.split("\n")
        lines = [self._MULTI_SPACE_RE.sub(" ", line).strip() for line in lines]
        text = "\n".join(lines)

        # 5. Collapse blank lines (more than 2 consecutive newlines → 2)
        text = self._MULTI_BLANK_LINE_RE.sub("\n\n", text)

        return text.strip()
