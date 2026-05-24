실행 예시:
python -m smartfarm.research_support.semantic_scholar_search --crop basil --crop-ko 바질 --limit-per-query 2 --max-queries-per-category 2 --save

결과 파일:
smartfarm/outputs/basil_source_bundle.json

역할:
작물명을 입력하면 카테고리별 쿼리를 자동 생성하고, Semantic Scholar API로 관련 논문을 검색해 source_bundle JSON으로 저장한다.