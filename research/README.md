실행 예시 (저장소 루트에서):
python -m research.research_support.semantic_scholar_search --crop basil --crop-ko 바질 --limit-per-query 2 --max-queries-per-category 2 --save

API 키(선택): 환경변수 SEMANTIC_SCHOLAR_API_KEY. 키가 없어도 research/cache/semantic_scholar/ 에 캐시된 쿼리(basil, strawberry; limit-per-query 2)는 네트워크 없이 재사용된다.

결과 파일:
research/outputs/basil_source_bundle.json

캐시 위치:
research/cache/semantic_scholar/

역할:
작물명을 입력하면 카테고리별 쿼리를 자동 생성하고, Semantic Scholar API로 관련 논문을 검색해 source_bundle JSON으로 저장한다.
