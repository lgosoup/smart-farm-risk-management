# Smart Farm Risk Management System - 자동 설정 스크립트 (PowerShell)
# 이 스크립트는 가상환경을 만들고 모든 필요한 패키지를 설치합니다.
# 
# 실행 방법:
#   PowerShell을 관리자 권한으로 열고:
#   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
#   .\setup.ps1

Write-Host ""
Write-Host "========================================"
Write-Host "Smart Farm Risk Management System 설정"
Write-Host "========================================"
Write-Host ""

# Python 버전 확인
Write-Host "Python 버전 확인:"
python --version
Write-Host ""

# 가상환경 생성
Write-Host "[1/3] 가상환경 생성 중..."
if (Test-Path ".venv") {
    Write-Host "가상환경이 이미 존재합니다. 스킵합니다."
} else {
    python -m venv .venv
    Write-Host "가상환경 생성 완료."
}
Write-Host ""

# 가상환경 활성화
Write-Host "[2/3] 가상환경 활성화 중..."
& ".venv\Scripts\Activate.ps1"
Write-Host ""

# pip 업그레이드
Write-Host "pip 업그레이드 중..."
python -m pip install --upgrade pip setuptools
Write-Host ""

# 패키지 설치
Write-Host "[3/3] 필요한 패키지 설치 중..."
Write-Host "이 과정은 시간이 걸릴 수 있습니다 (특히 torch 설치)."
pip install -r requirements.txt
Write-Host ""

Write-Host "========================================"
Write-Host "설정 완료!"
Write-Host "========================================"
Write-Host ""
Write-Host "다음에 가상환경을 활성화하려면:"
Write-Host "  & '.venv\Scripts\Activate.ps1'"
Write-Host ""
Write-Host "Streamlit 앱 실행:"
Write-Host "  streamlit run streamlit_app.py"
Write-Host ""
