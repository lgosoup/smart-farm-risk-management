# Smart Farm Risk Management System

수경재배 스마트팜의 수질·환경 센서값을 퍼지 로직으로 평가해 위험도를 판정하고, 그 결과를 바탕으로 장치 제어안을 자동으로 고르는 시스템입니다. 판정 결과는 로컬 LLM이 사람이 읽기 쉬운 설명으로 풀어 주고, 매일 한 번씩 밴딧 알고리즘이 퍼지 파라미터를 조금씩 조정해 작물 환경에 맞춰 갑니다.

## 작품 개요

스마트팜에서 수온, pH, EC, 유량, 탁도 같은 값은 하나만 보고 판단하기 어렵습니다. 경계값 근처의 값이 여러 개 겹치거나, 한 값이 서서히 나빠지는 추세도 함께 봐야 합니다. 이 프로젝트는 이런 판단을 퍼지 규칙으로 옮겨 0~100점의 위험 점수와 `안정 / 경계 / 주의 / 경고 / 위험 / 긴급` 단계로 나타냅니다.

판정이 `경고` 이상이면 펌프, 산소펌프, 펠티어 냉각기, 팬 같은 장치의 가동 조합을 후보로 만들고, 각 후보를 적용했을 때의 위험도 감소와 에너지·양액 사용량을 함께 따져 가장 알맞은 제어안을 고릅니다.

작물이 바뀌면 기준도 바뀌어야 하므로, 농촌진흥청 농사로 API 문서와 Semantic Scholar 논문에서 작물별 적정 범위를 뽑아 작물 프로필을 만들고, 이를 퍼지 멤버십 함수에 반영합니다. 현재 바질, 상추, 딸기, 토마토, 와사비 프로필이 들어 있습니다.

```
센서 입력 ─▶ 퍼지 위험 평가 ─▶ 경보 저장 / 상태 갱신
                 │
                 ├─▶ (경고 이상) 제어 최적화 ─▶ 장치 명령
                 ├─▶ (요청 시) LLM 해설
                 └─▶ (매일 03:00) 밴딧 기반 파라미터 개선 ◀─ 웹캠 식물 건강도
```

## 주요 기능

### 1. 퍼지 위험 평가 (`fuzzyLozic/`)
- 수온, pH, EC, 유량비, 탁도를 사다리꼴·삼각형 멤버십 함수로 퍼지화합니다. 기온과 습도는 모니터링용으로만 기록합니다.
- Mamdani 방식(min 함의, max 집계)으로 규칙을 평가하고, 무게중심법으로 위험 점수(0~100)를 계산합니다.
- EC는 `water`(일반 수질)와 `hydro`(양액) 두 모드로 기준을 나눠 판정합니다.
- 최근 입력 이력으로 추세, 변동성, 지속, 급변, 회복 실패를 감지하고 이를 추가 규칙으로 반영합니다.
- 센서값이 범위를 벗어나거나 급격히 튀면 무결성 플래그를 남깁니다(값을 임의로 자르지 않음).
- `위험`이 연속 3회 나오면 `긴급`으로 올립니다.
- 발화한 규칙, 주요 원인, 권장 조치를 담은 근거 패킷(evidence packet)을 만들어 이력으로 저장합니다.

### 2. 장치 제어 최적화 (`fuzzyLozic/control/`)
- 순환펌프·양액펌프·산소펌프·냉각기·팬 가동 시간의 조합을 후보로 만듭니다.
- 후보마다 다음 센서값을 예측하고 다시 퍼지 평가를 돌려 예상 위험 점수를 구합니다.
- 위험 감소량, 에너지(Wh), 양액 사용량, 직전 제어와의 변화량, 과잉 제어, 효과가 약한 제어를 가중합한 목적함수 J로 최적안을 고릅니다.
- 고른 제어안을 아두이노 핀 단위 명령(D5~D12)으로 바꿉니다. 펠티어는 냉각펌프·팬을 먼저 켜고 끈 뒤에도 일정 시간 유지하는 보호 시퀀스를 따릅니다.

### 3. 밴딧 기반 자기 개선 (`fuzzyLozic/improvement/`)
- 멤버십 경계 이동, 규칙 가중치 조정 같은 작은 수정안을 후보 풀로 둡니다.
- UCB1로 하루 하나의 수정안을 고르고, LLM 게이트(LLM이 없으면 보수적인 휴리스틱)가 적용 여부를 `APPROVE / HOLD`로 결정합니다.
- 보상은 웹캠으로 찍은 식물 사진의 건강도를 우선 사용하고, 사진이 없으면 최근 경보 비율과 위험 점수 안정성으로 대신합니다.
- 작물 설정의 `improvement_limits` 안에서만 조정해 기준이 한 번에 크게 흔들리지 않게 합니다.

### 4. 식물 건강도 분석 (`smartfarm/vision/`)
- ExG(Excess Green) 지수와 HSV 채도로 잎 색상 건강도를 계산합니다.
- 이전 날짜 사진과 비교한 변화 점수를 더해 하루 단위 복합 점수를 냅니다.

### 5. LLM 해설 (`llm/`)
- Qwen3-4B를 4bit(NF4) 양자화로 GPU에 올려 사용합니다.
- 퍼지 로직이 정한 상태는 그대로 두고, 근거 패킷과 최근 이력만 보고 요약·주요 원인·권장 조치·판정 근거를 JSON으로 출력합니다.
- 질문에 따라 설명 중심 또는 조치 중심으로 답합니다.

### 6. 작물 프로필 생성 (`smartfarm/`, `research/`)
- 농사로 작목별 농업기술정보 API에서 재배 문서를 받아 옵니다.
- 문장 분리 → 키워드 필터 → 수치·단위 추출 → 문맥 분류 순서의 NLP 파이프라인으로 적정 수온, pH, EC 범위를 뽑습니다.
- Semantic Scholar 논문 검색 결과를 함께 모아 근거 묶음(source bundle)으로 저장합니다.
- 추출한 값을 검증·단위 정규화해 작물 프로필로 만들고, 퍼지 멤버십과 제어 목표값으로 변환해 `data/crop_config.json`에 적용합니다.

### 7. 오케스트레이터와 대시보드
- `orchestrator.py`: 60분마다 센서 수집 → 퍼지 평가 → 상태·경보 저장 → 제어 최적화 → LLM 요청 처리를 반복하고, 매일 03:00 이후 첫 사이클에 밴딧 개선을 실행합니다.
- 센서는 어댑터 구조로 분리되어 있어, `inputs.json`을 읽는 기본 어댑터를 실제 센서 수집 스크립트 어댑터로 바꿔 끼울 수 있습니다.
- `streamlit_app.py`: 입력값을 바로 바꿔 가며 멤버십, 발화 규칙, 시계열·무결성 플래그, 위험도 결과를 확인하는 데모 화면입니다.

### 8. 넷제로 확장 (`netzero/`)
- **기능 중심 장치 추상화:** 장치를 이름이 아니라 수행 기능, 제어 단계, 소비 전력, 환경 영향으로 표현합니다. 제어 후보와 핀 단위 명령은 시설 프로필에서 자동으로 만들어지므로, 장치 구성이 바뀌어도 코드를 고치지 않습니다.
- **학습형 환경 예측:** 장치 작동 전후의 실제 변화로 시설별 열 손실, 장치별 냉각 효과 등을 재귀 최소제곱(RLS)으로 계속 갱신합니다.
- **상시 최적화:** 경고 때만이 아니라 매 주기 최적화합니다. 먼저 적정 생육 범위를 지키는 후보를 고르고, 그 안에서 에너지, 탄소, 양액, 장치 변경을 최소화합니다.
- **탄소·요금 반영:** 시간대별 전력 배출계수와 계시별 요금을 에너지 비용에 반영합니다.
- **가상 스마트팜과 비교 실험:** 시설의 실제 물리 특성을 제어기에 숨긴 시뮬레이터에서 고정 임계값 제어(A), 기존 방식(B), 상시 최적화(C), 탄소 반영(D)을 같은 조건으로 비교합니다.

## 사용 기술

| 구분 | 내용 |
|---|---|
| 언어 | Python 3.10+ |
| 퍼지 추론 | 직접 구현한 Mamdani 추론 엔진 (사다리꼴·삼각형 멤버십, 무게중심법) |
| 최적화·학습 | 후보 탐색 기반 다목적 최적화, UCB1 멀티암드 밴딧, 재귀 최소제곱(RLS) 시스템 식별 |
| 시뮬레이션 | 시설·기후·전력망 프로필 기반 가상 스마트팜 |
| LLM | Qwen3-4B, Hugging Face Transformers, bitsandbytes 4bit 양자화, PyTorch (CUDA 11.8) |
| 영상 처리 | OpenCV, NumPy (ExG 지수, HSV 채도) |
| 데이터 수집 | 농촌진흥청 농사로 API, Semantic Scholar API, requests |
| 자연어 처리 | 한국어 농업 문서용 규칙 기반 수치 추출 파이프라인 |
| UI | Streamlit, Matplotlib |
| 하드웨어 | Arduino 핀 제어 (BTS7960 펌프 드라이버, 펠티어 냉각기, 팬, LED) |
| 테스트 | pytest |

## 프로젝트 구조

```
├── orchestrator.py          # 주기 실행 메인 루프
├── app.py                   # 퍼지 평가 + LLM 해설 1회 실행 (CLI)
├── streamlit_app.py         # 퍼지 로직 데모 대시보드
├── fuzzyLozic/
│   ├── config.py            # 멤버십 함수·출력 집합·정책 기본값
│   ├── fuzzify.py           # 퍼지화, 시계열·무결성 플래그
│   ├── rules.py             # 규칙 베이스
│   ├── inference.py         # Mamdani 추론, 무게중심법
│   ├── explain.py           # 근거 패킷 생성
│   ├── main.py              # 평가 실행 및 이력 저장
│   ├── control/             # 제어 후보 생성, 예측, 점수, 장치 명령
│   ├── improvement/         # 밴딧, LLM 게이트, 보상, 파라미터 적용
│   └── tests/
├── llm/                     # 모델 로더, 프롬프트, 응답 생성
├── smartfarm/
│   ├── public_api/          # 농사로 API 클라이언트
│   ├── document_nlp/        # 문서 수치 추출 파이프라인
│   ├── crop_profile/        # 작물 프로필 생성·검증·저장 (profiles/*.json)
│   ├── fuzzy_config/        # 작물 프로필 → 퍼지 설정 적용
│   ├── vision/              # 식물 건강도 분석
│   ├── generate_crop_profile.py
│   └── activate_crop.py
├── research/                # Semantic Scholar 논문 검색, 결과 캐시
├── netzero/                 # 넷제로 확장: 장치 추상화, 학습형 예측, 탄소 반영 최적화, 가상 스마트팜
│   ├── facility.py          # 기능 중심 장치 프로필, 제어 후보 생성, 장치 명령 변환
│   ├── model.py             # 장치 작동 기록으로 갱신되는 환경 예측 모델 (RLS)
│   ├── controllers.py       # A 고정 임계값 / B 기존 방식 / C 상시 최적화 / D 탄소·요금 반영
│   ├── simulator.py         # 시설 특성을 숨긴 가상 스마트팜, 기후 프로필
│   ├── grid.py              # 시간대별 전력 배출계수·요금
│   ├── experiment.py        # 비교 실험 실행과 지표 계산
│   └── profiles/            # facilities/, climate/, grid/ (JSON)
├── netzero_app.py           # 넷제로 비교 실험 시연 화면
├── tests/                   # 실행 경로(오케스트레이터, app, LLM, 대시보드) 테스트
└── data/                    # 입력, 작물 설정, 상태, 출력 파일
```

## 설치 및 실행

### 설치
```bash
# Windows 자동 설정 (가상환경 생성 + 패키지 설치)
setup.bat
# 또는 PowerShell
.\setup.ps1

# 직접 설치
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# LLM 기능을 쓸 때만 (NVIDIA GPU 필요): setup.bat gpu  또는
pip install -r requirements-gpu.txt --extra-index-url https://download.pytorch.org/whl/cu118
```

LLM 패키지가 없거나 GPU가 없으면 LLM 해설과 LLM 게이트는 퍼지 결과 요약과 휴리스틱 판단으로 대신 동작합니다.

### 실행
```bash
# 대시보드
streamlit run streamlit_app.py

# 주기 실행 (60분 간격, Ctrl+C로 종료)
python orchestrator.py

# 퍼지 평가 + LLM 해설 1회
python app.py "현재 상태를 근거와 함께 자세히 설명"
```

### 넷제로 비교 실험
```bash
# 시설 3종 × 제어 방식 A~D, 7일 시뮬레이션
python -m netzero.experiment --days 7

# 폭염 + 3일차 칠러 성능 50% 저하
python -m netzero.experiment --facility mid_greenhouse --climate heatwave --event degrade:chiller_a:0.5@3

# 작물 변경 + 2일차 환풍기 제거
python -m netzero.experiment --facility small_rack --crop strawberry --event remove:vent_fan@2

# 시연 화면
streamlit run netzero_app.py
```

새 시설은 `netzero/profiles/facilities/`에 장치 프로필 JSON을 추가하면 됩니다. 장치마다 수행 기능, 제어 단계, 소비 전력, 환경 영향의 초기 추정값(prior)을 적고, 실제 영향은 운전하면서 학습합니다. `simulator_truth` 블록은 가상 스마트팜만 읽는 숨은 물리 특성입니다.

### 테스트
```bash
python -m pytest -q fuzzyLozic/tests smartfarm/tests tests netzero/tests
```

### 작물 설정
```bash
# 등록된 작물 목록과 현재 작물 확인
python -m smartfarm.generate_crop_profile --list
python -m smartfarm.generate_crop_profile --status

# 작물 프로필 생성 후 적용 (농사로 API 키 필요)
python -m smartfarm.generate_crop_profile --crop 바질 --activate

# 이미 있는 프로필로 바로 전환
python -m smartfarm.activate_crop basil
```

### 환경 변수
| 이름 | 용도 |
|---|---|
| `NONGSARO_API_KEY` | 농사로 API 서비스 키 (공공데이터포털 발급) |
| `SEMANTIC_SCHOLAR_API_KEY` | Semantic Scholar API 키 (선택, 없으면 기본 요청 한도로 동작) |

## 데이터 파일

| 파일 | 내용 |
|---|---|
| `data/inputs.json` | 현재 센서값 (수온, pH, EC, 유량비, 탁도, 기온, 습도) |
| `data/crop_config.json` | 적용 중인 작물, 제어 목표값, 멤버십 재정의 |
| `data/outputs.json` | 최근 퍼지 평가 결과 |
| `data/device_control_outputs.json` | 최근 제어 최적화 결과와 장치 명령 |
| `data/improvement_state.json` | 밴딧 통계와 개선된 파라미터 |

실행 중에는 `history_*.jsonl`, `alerts.jsonl`, `system_status.json`, `llm_request.json` / `llm_response.json` 같은 파일이 `data/`에 생기며, 웹 화면과 주고받는 용도로 쓰입니다.
