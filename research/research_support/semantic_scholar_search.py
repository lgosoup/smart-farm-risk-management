from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


SEMANTIC_SCHOLAR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

OUTPUT_DIR = Path("smartfarm/outputs")
CACHE_DIR = Path("smartfarm/cache/semantic_scholar")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. 작물 기본 정보
# ============================================================

CROP_PRESETS: Dict[str, Dict[str, Optional[str]]] = {
    "basil": {
        "crop_ko": "바질",
        "scientific_name": "Ocimum basilicum",
    },
    "lettuce": {
        "crop_ko": "상추",
        "scientific_name": "Lactuca sativa",
    },
    "strawberry": {
        "crop_ko": "딸기",
        "scientific_name": "Fragaria × ananassa",
    },
    "tomato": {
        "crop_ko": "토마토",
        "scientific_name": "Solanum lycopersicum",
    },
    "wasabi": {
        "crop_ko": "와사비",
        "scientific_name": "Eutrema japonicum",
    },
}


def normalize_crop(
    crop: str,
    crop_ko: Optional[str] = None,
    scientific_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    작물명을 내부에서 사용할 표준 형태로 정리한다.
    """
    crop_en = crop.strip().lower()
    preset = CROP_PRESETS.get(crop_en, {})

    return {
        "crop": crop_en.replace(" ", "_"),
        "crop_common_name_en": crop_en,
        "crop_common_name_ko": crop_ko or preset.get("crop_ko") or crop,
        "scientific_name": scientific_name or preset.get("scientific_name"),
    }


# ============================================================
# 2. 카테고리별 자동 쿼리 생성
# ============================================================

QUERY_TEMPLATES: Dict[str, List[str]] = {
    "water_temp": [
        "{crop} hydroponic water temperature range",
        "{crop} nutrient solution temperature hydroponic",
        "{sci} hydroponic water temperature",
    ],
    "pH": [
        "{crop} hydroponic pH range",
        "{crop} nutrient solution pH optimal",
        "{sci} hydroponic pH nutrient solution",
    ],
    "EC": [
        "{crop} hydroponic EC range",
        "{crop} nutrient solution electrical conductivity",
        "{sci} hydroponic electrical conductivity",
    ],
    "flow": [
        "{crop} hydroponic flow rate",
        "{crop} hydroponic water circulation",
        "{crop} hydroponic nutrient solution flow",
    ],
    "turbidity": [
        "{crop} hydroponic water quality turbidity",
        "{crop} root disease water quality hydroponic",
        "{crop} hydroponic nutrient solution clarity",
    ],
    "DO": [
        "{crop} hydroponic dissolved oxygen",
        "{crop} root oxygen hydroponic",
        "{sci} hydroponic dissolved oxygen root",
    ],
    "humidity": [
        "{crop} growing relative humidity",
        "{crop} greenhouse humidity range",
        "{sci} humidity plant growth",
    ],
    "light": [
        "{crop} hydroponic light requirement",
        "{crop} PPFD DLI photoperiod hydroponic",
        "{sci} light requirement hydroponic",
    ],
    "crop_risk": [
        "{crop} root rot hydroponic",
        "{crop} bolting temperature hydroponic",
        "{crop} disease hydroponic nutrient solution",
        "{sci} root disease hydroponic",
    ],
    "KR": [
        "{crop_ko} 수경재배 적정 온도 pH EC 양액",
        "{crop_ko} 수경재배 양액 기준",
        "{crop_ko} 수경재배 생육 단계별 양액 농도",
        "{crop_ko} 수경재배 병해 뿌리",
    ],
}


def build_query_plan(crop_info: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    작물 정보에 따라 카테고리별 검색 쿼리를 자동 생성한다.
    """
    crop = crop_info["crop_common_name_en"]
    crop_ko = crop_info["crop_common_name_ko"]
    sci = crop_info.get("scientific_name") or crop

    query_plan: Dict[str, List[str]] = {}

    for category, templates in QUERY_TEMPLATES.items():
        queries = []

        for template in templates:
            query = template.format(
                crop=crop,
                crop_ko=crop_ko,
                sci=sci,
            )

            # scientific_name이 없을 때 crop과 같은 중복성 쿼리 제거
            if query not in queries:
                queries.append(query)

        query_plan[category] = queries

    return query_plan


# ============================================================
# 3. 유틸 함수
# ============================================================

def safe_filename(text: str, max_len: int = 80) -> str:
    text = text.lower().strip()
    safe = "".join(ch if ch.isalnum() else "_" for ch in text)
    safe = "_".join(part for part in safe.split("_") if part)

    if len(safe) > max_len:
        safe = safe[:max_len]

    return safe or "result"


def cache_key(category: str, query: str, limit: int, year: Optional[str]) -> str:
    raw = f"{category}|{query}|limit={limit}|year={year}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def cache_path(category: str, query: str, limit: int, year: Optional[str]) -> Path:
    key = cache_key(category, query, limit, year)
    name = f"{safe_filename(category)}_{key}.json"
    return CACHE_DIR / name


def extract_pdf_url(open_access_pdf: Any) -> Optional[str]:
    if isinstance(open_access_pdf, dict):
        return open_access_pdf.get("url")

    if isinstance(open_access_pdf, str):
        return open_access_pdf

    return None


def normalize_document(
    item: Dict[str, Any],
    category: str,
    query: str,
) -> Optional[Dict[str, Any]]:
    title = item.get("title")

    if not title:
        return None

    external_ids = item.get("externalIds") or {}

    return {
        "category": category,
        "query": query,

        "title": title,
        "abstract": item.get("abstract"),
        "url": item.get("url"),
        "openAccessPdf": extract_pdf_url(item.get("openAccessPdf")),
        "source_type": "paper",

        "paperId": item.get("paperId"),
        "year": item.get("year"),
        "venue": item.get("venue"),
        "citationCount": item.get("citationCount"),
        "doi": external_ids.get("DOI"),
    }


def deduplicate_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    같은 논문이 여러 카테고리에서 중복 검색될 수 있으므로 중복 제거한다.
    단, 어떤 카테고리/쿼리에서 검색됐는지 정보는 유지하기 위해 matched_queries에 누적한다.
    """
    by_key: Dict[str, Dict[str, Any]] = {}

    for doc in documents:
        key = doc.get("paperId") or (doc.get("title") or "").strip().lower()

        if not key:
            continue

        current_match = {
            "category": doc.get("category"),
            "query": doc.get("query"),
        }

        if key not in by_key:
            new_doc = dict(doc)
            new_doc["matched_queries"] = [current_match]
            by_key[key] = new_doc
        else:
            by_key[key]["matched_queries"].append(current_match)

    return list(by_key.values())


# ============================================================
# 4. Semantic Scholar API 호출
# ============================================================

def load_cache(
    category: str,
    query: str,
    limit: int,
    year: Optional[str],
) -> Optional[List[Dict[str, Any]]]:
    path = cache_path(category, query, limit, year)

    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("documents", [])
    except Exception:
        return None


def save_cache(
    category: str,
    query: str,
    limit: int,
    year: Optional[str],
    documents: List[Dict[str, Any]],
) -> None:
    path = cache_path(category, query, limit, year)

    payload = {
        "category": category,
        "query": query,
        "limit": limit,
        "year": year,
        "documents": documents,
        "cached_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def search_semantic_scholar(
    category: str,
    query: str,
    limit: int = 3,
    api_key: Optional[str] = None,
    year: Optional[str] = None,
    use_cache: bool = True,
    max_retries: int = 3,
    base_wait_sec: int = 20,
) -> Dict[str, Any]:
    """
    단일 쿼리로 Semantic Scholar 검색을 수행한다.
    """
    if use_cache:
        cached = load_cache(category, query, limit, year)
        if cached is not None:
            return {
                "status": "ok_cached",
                "documents": cached,
                "message": "Loaded from cache.",
            }

    fields = ",".join(
        [
            "paperId",
            "title",
            "abstract",
            "year",
            "url",
            "openAccessPdf",
            "externalIds",
            "citationCount",
            "venue",
        ]
    )

    params: Dict[str, Any] = {
        "query": query,
        "limit": limit,
        "fields": fields,
    }

    if year:
        params["year"] = year

    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    last_error = ""

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(
                SEMANTIC_SCHOLAR_SEARCH_URL,
                params=params,
                headers=headers,
                timeout=20,
            )
        except requests.RequestException as exc:
            last_error = str(exc)
            wait_time = base_wait_sec * attempt
            print(f"[WARN] 요청 실패: {last_error}")
            print(f"[WARN] {wait_time}초 대기 후 재시도합니다. ({attempt}/{max_retries})")
            time.sleep(wait_time)
            continue

        if response.status_code == 200:
            data = response.json()
            documents: List[Dict[str, Any]] = []

            for item in data.get("data", []):
                doc = normalize_document(
                    item=item,
                    category=category,
                    query=query,
                )
                if doc is not None:
                    documents.append(doc)

            if use_cache:
                save_cache(category, query, limit, year, documents)

            return {
                "status": "ok",
                "documents": documents,
                "message": "Search succeeded.",
            }

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                wait_time = int(retry_after)
            else:
                wait_time = base_wait_sec * attempt

            last_error = response.text[:300]
            print(f"[WARN] 429 Too Many Requests: {wait_time}초 대기 후 재시도합니다. ({attempt}/{max_retries})")
            time.sleep(wait_time)
            continue

        last_error = response.text[:300]
        return {
            "status": "error",
            "documents": [],
            "message": f"API error {response.status_code}: {last_error}",
        }

    return {
        "status": "rate_limited",
        "documents": [],
        "message": f"Request failed after retries. Last error: {last_error}",
    }


# ============================================================
# 5. 작물별 source bundle 생성
# ============================================================

def collect_crop_source_bundle(
    crop: str,
    crop_ko: Optional[str] = None,
    scientific_name: Optional[str] = None,
    limit_per_query: int = 2,
    max_queries_per_category: int = 2,
    year: Optional[str] = None,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """
    작물명 하나를 입력받아 여러 카테고리 쿼리를 자동 생성하고,
    모든 검색 결과를 하나의 source_bundle로 묶는다.
    """
    crop_info = normalize_crop(
        crop=crop,
        crop_ko=crop_ko,
        scientific_name=scientific_name,
    )

    query_plan = build_query_plan(crop_info)

    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

    all_documents: List[Dict[str, Any]] = []
    query_results: List[Dict[str, Any]] = []

    print("[INFO] Crop information")
    print(json.dumps(crop_info, ensure_ascii=False, indent=2))

    for category, queries in query_plan.items():
        print(f"\n[CATEGORY] {category}")

        selected_queries = queries[:max_queries_per_category]

        for query in selected_queries:
            print(f"  - query: {query}")

            result = search_semantic_scholar(
                category=category,
                query=query,
                limit=limit_per_query,
                api_key=api_key,
                year=year,
                use_cache=use_cache,
            )

            docs = result["documents"]
            all_documents.extend(docs)

            query_results.append(
                {
                    "category": category,
                    "query": query,
                    "status": result["status"],
                    "message": result["message"],
                    "document_count": len(docs),
                }
            )

            # API Key가 있어도 너무 빠르게 호출하지 않도록 대기
            time.sleep(1.2)

    deduped_documents = deduplicate_documents(all_documents)

    bundle = {
        "bundle_type": "source_bundle",
        "crop_info": crop_info,
        "query_plan": query_plan,
        "search_config": {
            "limit_per_query": limit_per_query,
            "max_queries_per_category": max_queries_per_category,
            "year": year,
            "use_cache": use_cache,
        },
        "search_summary": {
            "total_raw_documents": len(all_documents),
            "total_unique_documents": len(deduped_documents),
            "category_count": len(query_plan),
            "query_count_executed": len(query_results),
        },
        "query_results": query_results,
        "documents": deduped_documents,
        "next_step": {
            "target_module": "DocumentNLPExtractor",
            "description": (
                "documents 배열의 title, abstract, openAccessPdf를 이용해 "
                "수온, pH, EC, 유량, 탁도, DO, 습도, 광량, 작물 특이 위험 관련 "
                "파라미터 후보를 추출한다."
            ),
        },
    }

    return bundle


def save_source_bundle(bundle: Dict[str, Any]) -> Path:
    crop = bundle["crop_info"]["crop"]
    filename = f"{safe_filename(crop)}_source_bundle.json"
    output_path = OUTPUT_DIR / filename

    output_path.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return output_path


# ============================================================
# 6. 실행
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crop input -> automatic query generation -> Semantic Scholar source bundle"
    )

    parser.add_argument("--crop", required=True, help="영문 작물명. 예: basil")
    parser.add_argument("--crop-ko", default=None, help="한글 작물명. 예: 바질")
    parser.add_argument("--scientific-name", default=None, help="학명. 예: Ocimum basilicum")
    parser.add_argument("--limit-per-query", type=int, default=2, help="쿼리당 검색 논문 개수")
    parser.add_argument("--max-queries-per-category", type=int, default=2, help="카테고리당 실행할 쿼리 개수")
    parser.add_argument("--year", default=None, help='연도 필터. 예: "2015-" 또는 "2020-2026"')
    parser.add_argument("--no-cache", action="store_true", help="캐시 사용 안 함")
    parser.add_argument("--save", action="store_true", help="source_bundle JSON 저장")

    args = parser.parse_args()

    bundle = collect_crop_source_bundle(
        crop=args.crop,
        crop_ko=args.crop_ko,
        scientific_name=args.scientific_name,
        limit_per_query=args.limit_per_query,
        max_queries_per_category=args.max_queries_per_category,
        year=args.year,
        use_cache=not args.no_cache,
    )

    print("\n[RESULT]")
    print(json.dumps(bundle, ensure_ascii=False, indent=2))

    if args.save:
        output_path = save_source_bundle(bundle)
        print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()