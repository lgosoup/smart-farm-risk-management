@echo off
REM Smart Farm Risk Management System - 자동 설정 스크립트 (Windows)
REM 이 스크립트는 가상환경을 만들고 모든 필요한 패키지를 설치합니다.

echo.
echo ========================================
echo Smart Farm Risk Management System 설정
echo ========================================
echo.

REM Python 버전 확인
python --version
echo.

REM 가상환경 생성
echo [1/3] 가상환경 생성 중...
if exist .venv (
    echo 가상환경이 이미 존재합니다. 스킵합니다.
) else (
    python -m venv .venv
    echo 가상환경 생성 완료.
)
echo.

REM 가상환경 활성화
echo [2/3] 가상환경 활성화 중...
call .venv\Scripts\activate.bat
echo.

REM pip 업그레이드
echo pip 업그레이드 중...
python -m pip install --upgrade pip setuptools
echo.

REM 패키지 설치
echo [3/3] 필요한 패키지 설치 중...
echo 이 과정은 시간이 걸릴 수 있습니다 (특히 torch 설치).
pip install -r requirements.txt
echo.

echo ========================================
echo 설정 완료!
echo ========================================
echo.
echo 다음에 가상환경을 활성화하려면:
echo   .venv\Scripts\activate.bat
echo.
echo Streamlit 앱 실행:
echo   streamlit run streamlit_app.py
echo.
pause
