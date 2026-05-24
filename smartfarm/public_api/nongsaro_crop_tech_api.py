"""농촌진흥청 작목별농업기술정보 API client.

Wraps the Nongsaro (농사로) public REST/XML API and exposes a clean
:class:`NongsaroCropTechApiClient` interface for fetching crop-technology
documents by crop name.

Environment variables
---------------------
NONGSARO_API_KEY
    Service key issued by 공공데이터포털 (data.go.kr).
    Required for live requests; if absent the client returns an empty result
    with an explanatory error message instead of raising an exception.

Caching
-------
Responses are cached as JSON files under *cache_dir* (default:
``smartfarm/cache/nongsaro/``).  Cache filenames are MD5 hashes of the
query string so filenames stay filesystem-safe regardless of the script's
working directory or input encoding.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

try:
    import requests
    from requests.exceptions import RequestException
except ImportError as _req_err:  # pragma: no cover
    raise ImportError(
        "The 'requests' library is required. Install it with: pip install requests"
    ) from _req_err

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE_URL = "http://api.nongsaro.go.kr/service/cropDiseaseService"
_PRDLST_URL = "http://api.nongsaro.go.kr/service/prdlst"

# Endpoint paths
_ENDPOINT_LIST = "/cropTechList"
_ENDPOINT_DETAIL = "/cropTechDetail"

# Timeout for HTTP requests (seconds)
_REQUEST_TIMEOUT: int = 15

# Number of rows to request per page
_NUM_OF_ROWS: int = 10

# Seconds to wait between successive API calls to avoid rate-limiting
_INTER_REQUEST_DELAY: float = 0.3

# Default cache directory relative to this file's package root
_DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "nongsaro"


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class NongsaroCropTechApiClient:
    """Fetch crop-technology documents from the 농촌진흥청 Nongsaro API.

    Parameters
    ----------
    api_key:
        Service key for the Nongsaro open API.  If *None*, the value of the
        ``NONGSARO_API_KEY`` environment variable is used.  An empty string
        is treated as a missing key.
    cache_dir:
        Directory for on-disk JSON response caches.  Created automatically
        if it does not exist.  Pass ``None`` to disable caching.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.api_key: str = (
            api_key
            if api_key is not None
            else os.environ.get("NONGSARO_API_KEY", "")
        )

        if cache_dir is None:
            self._cache_dir: Optional[Path] = _DEFAULT_CACHE_DIR
        elif cache_dir == "":
            self._cache_dir = None  # caching disabled
        else:
            self._cache_dir = Path(cache_dir)

        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/xml, text/xml, */*"})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_crop_documents(self, crop_info: dict) -> dict:
        """Fetch crop-technology documents for the given normalized crop profile.

        The method tries multiple search strategies in order:

        1. Korean common name (``crop_common_name_ko``)
        2. English common name (``crop_common_name_en``)

        Parameters
        ----------
        crop_info:
            Normalized crop profile produced by
            :func:`smartfarm.crop_input.crop_input_normalizer.normalize`.

        Returns
        -------
        dict
            ``{"crop": str, "documents": list[dict], "search_queries": list,
            "api_calls": int}``

            Each document has keys: ``title``, ``source``, ``raw_text``,
            ``url``, ``source_type``.

            On failure an additional ``"error"`` key and ``"fallback": True``
            are included.
        """
        if not self.api_key:
            logger.warning("NONGSARO_API_KEY is not set; returning empty result.")
            return {
                "crop": crop_info.get("crop_common_name_ko")
                        or crop_info.get("crop_common_name_en", "unknown"),
                "documents": [],
                "search_queries": [],
                "api_calls": 0,
                "error": (
                    "API key is missing.  Set the NONGSARO_API_KEY environment "
                    "variable or pass api_key to NongsaroCropTechApiClient()."
                ),
                "fallback": True,
            }

        crop_label = (
            crop_info.get("crop_common_name_ko")
            or crop_info.get("crop_common_name_en")
            or "unknown"
        )
        queries_tried: list[str] = []
        api_calls: int = 0
        all_documents: list[dict] = []

        for field in ("crop_common_name_ko", "crop_common_name_en"):
            query = crop_info.get(field)
            if not query:
                continue
            queries_tried.append(query)
            docs, calls = self._search_by_name(query)
            api_calls += calls
            all_documents.extend(docs)
            time.sleep(_INTER_REQUEST_DELAY)
            if all_documents:
                break  # first successful strategy wins

        result: dict = {
            "crop": crop_label,
            "documents": all_documents,
            "search_queries": queries_tried,
            "api_calls": api_calls,
        }
        if not all_documents:
            result["fallback"] = True
        return result

    # ------------------------------------------------------------------
    # Internal search and parsing helpers
    # ------------------------------------------------------------------

    def _search_by_name(self, query: str) -> tuple[list[dict], int]:
        """Search the API for *query* and return ``(documents, api_call_count)``.

        On any error an empty list is returned and the error is logged.
        """
        cache_key = self._cache_key(query)
        cached = self._load_cache(cache_key)
        if cached is not None:
            logger.debug("Cache hit for query %r", query)
            return cached.get("documents", []), 0

        params = {
            "apiKey": self.api_key,
            "numOfRows": _NUM_OF_ROWS,
            "pageNo": 1,
            "sKwd": query,
            "format": "json",
        }

        # Primary endpoint
        url = _BASE_URL + _ENDPOINT_LIST

        try:
            response = self._session.get(url, params=params,
                                         timeout=_REQUEST_TIMEOUT)
            response.raise_for_status()
        except RequestException as exc:
            logger.error("HTTP error querying Nongsaro API for %r: %s", query, exc)
            return [], 1

        raw_content = response.text

        # The API may return JSON or XML depending on the endpoint and key.
        documents: list[dict] = []
        if raw_content.lstrip().startswith("<"):
            documents = self._parse_xml_response(raw_content, query)
        else:
            documents = self._parse_json_response(raw_content, query)

        payload = {"documents": documents}
        self._save_cache(cache_key, payload)
        return documents, 1

    def _parse_xml_response(self, xml_text: str, query: str = "") -> list[dict]:
        """Parse a Nongsaro XML response and return a list of document dicts."""
        documents: list[dict] = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            logger.error("XML parse error: %s", exc)
            return documents

        # Check resultCode inside <header>
        header = root.find(".//header")
        if header is not None:
            result_code = (header.findtext("resultCode") or "").strip()
            result_msg = (header.findtext("resultMsg") or "").strip()
            if result_code not in ("00", "0000", ""):
                logger.warning(
                    "Nongsaro API returned non-OK result code %r (%s) for query %r",
                    result_code, result_msg, query,
                )
                return documents

        # Items may live under different tag names depending on the endpoint
        for item in root.iter("item"):
            doc = self._item_element_to_document(item)
            if doc:
                documents.append(doc)

        return documents

    def _parse_json_response(self, json_text: str, query: str = "") -> list[dict]:
        """Parse a Nongsaro JSON response and return a list of document dicts."""
        documents: list[dict] = []
        try:
            data = json.loads(json_text)
        except json.JSONDecodeError as exc:
            logger.error("JSON parse error: %s", exc)
            return documents

        # Navigate common Nongsaro response envelope
        body = (
            data.get("response", {}).get("body", {})
            or data.get("body", {})
            or data
        )
        items_container = body.get("items") or body.get("item") or []
        if isinstance(items_container, dict):
            items_list = items_container.get("item", [])
        elif isinstance(items_container, list):
            items_list = items_container
        else:
            items_list = []

        for item in items_list:
            if not isinstance(item, dict):
                continue
            doc = self._item_dict_to_document(item)
            if doc:
                documents.append(doc)

        return documents

    # ------------------------------------------------------------------
    # Document-building helpers
    # ------------------------------------------------------------------

    def _item_element_to_document(self, item: ET.Element) -> Optional[dict]:
        """Convert an ``<item>`` XML element to a document dict."""
        title = (
            item.findtext("cntntsNm")
            or item.findtext("title")
            or item.findtext("cropTechNm")
            or ""
        ).strip()
        content = (
            item.findtext("cntnts")
            or item.findtext("content")
            or item.findtext("cropTechCntnts")
            or ""
        ).strip()
        cntnts_no = (item.findtext("cntntsNo") or "").strip()

        if not title and not content:
            return None

        url = self._build_detail_url(cntnts_no) if cntnts_no else _BASE_URL
        raw_text = self._strip_html(content) if content else ""

        return {
            "title": title or "(제목 없음)",
            "source": "농촌진흥청_작목별농업기술정보 API",
            "raw_text": raw_text,
            "url": url,
            "source_type": "government",
        }

    def _item_dict_to_document(self, item: dict) -> Optional[dict]:
        """Convert a JSON item dict to a document dict."""
        title = (
            item.get("cntntsNm")
            or item.get("title")
            or item.get("cropTechNm")
            or ""
        ).strip()
        content = (
            item.get("cntnts")
            or item.get("content")
            or item.get("cropTechCntnts")
            or ""
        ).strip()
        cntnts_no = str(item.get("cntntsNo", "")).strip()

        if not title and not content:
            return None

        url = self._build_detail_url(cntnts_no) if cntnts_no else _BASE_URL
        raw_text = self._strip_html(content) if content else ""

        return {
            "title": title or "(제목 없음)",
            "source": "농촌진흥청_작목별농업기술정보 API",
            "raw_text": raw_text,
            "url": url,
            "source_type": "government",
        }

    def _build_detail_url(self, cntnts_no: str) -> str:
        """Build a detail URL for a content ID."""
        return (
            f"{_BASE_URL}{_ENDPOINT_DETAIL}"
            f"?apiKey={self.api_key}&cntntsNo={cntnts_no}"
        )

    # ------------------------------------------------------------------
    # Text cleaning
    # ------------------------------------------------------------------

    def _strip_html(self, text: str) -> str:
        """Remove HTML tags and collapse whitespace from *text*."""
        # Remove HTML tags
        no_tags = re.sub(r"<[^>]+>", " ", text)
        # Decode common HTML entities
        no_tags = (
            no_tags
            .replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&nbsp;", " ")
            .replace("&quot;", '"')
            .replace("&#39;", "'")
        )
        # Collapse whitespace
        cleaned = re.sub(r"\s+", " ", no_tags).strip()
        return cleaned

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------

    def _cache_key(self, query: str) -> str:
        """Return a hex-string MD5 hash to use as the cache filename stem."""
        return hashlib.md5(query.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Optional[Path]:
        if self._cache_dir is None:
            return None
        return self._cache_dir / f"{key}.json"

    def _load_cache(self, key: str) -> Optional[dict]:
        """Return cached data for *key*, or ``None`` if not cached."""
        path = self._cache_path(key)
        if path is None or not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to read cache file %s: %s", path, exc)
            return None

    def _save_cache(self, key: str, data: dict) -> None:
        """Persist *data* to the cache directory under *key*."""
        path = self._cache_path(key)
        if path is None:
            return
        try:
            with path.open("w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2)
        except OSError as exc:
            logger.warning("Failed to write cache file %s: %s", path, exc)
