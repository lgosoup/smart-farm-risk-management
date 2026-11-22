# Smart Farm Risk Management System - 자동 설정 스크립트 (PowerShell)
# 이 스크립트는 가상환경을 만들고 모든 필요한 패키지를 설치합니다.
#
# 실행 방법 (관리자 권한 불필요):
#   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
#   .\setup.ps1          # 기본 패키지만
#   .\setup.ps1 -Gpu     # LLM용 GPU 패키지까지 (NVIDIA GPU 필요)
param([switch]$Gpu)

$ErrorActionPreference = "Stop"
# 어디서 실행하든 스크립트가 있는 폴더(저장소 루트) 기준으로 동작
Set-Location -Path $PSScriptRoot

function Invoke-Checked([string]$What, [scriptblock]$Cmd) {
    & $Cmd
    if ($LASTEXITCODE -ne 0) { throw "$What 실패 (exit $LASTEXITCODE)" }
}

Write-Host ""
Write-Host "========================================"
Write-Host "Smart Farm Risk Management System 설정"
Write-Host "========================================"
Write-Host ""

# Python 버전 확인
Write-Host "Python 버전 확인:"
Invoke-Checked "python --version" { python --version }
Write-Host ""

# 가상환경 생성
Write-Host "[1/3] 가상환경 생성 중..."
if (Test-Path ".venv\Scripts\python.exe") {
    Write-Host "가상환경이 이미 존재합니다. 스킵합니다."
} else {
    Invoke-Checked "가상환경 생성" { python -m venv .venv }
    Write-Host "가상환경 생성 완료."
}
Write-Host ""

# 가상환경 활성화
Write-Host "[2/3] 가상환경 활성화 중..."
& ".venv\Scripts\Activate.ps1"
Write-Host ""

# pip 업그레이드
Write-Host "pip 업그레이드 중..."
Invoke-Checked "pip 업그레이드" { python -m pip install --upgrade pip setuptools }
Write-Host ""

# 패키지 설치
Write-Host "[3/3] 필요한 패키지 설치 중..."
Invoke-Checked "requirements.txt 설치" { python -m pip install -r requirements.txt }
if ($Gpu) {
    Write-Host "LLM용 GPU 패키지 설치 중 (torch 등, 시간이 오래 걸립니다)..."
    Invoke-Checked "requirements-gpu.txt 설치" {
        python -m pip install -r requirements-gpu.txt --extra-index-url https://download.pytorch.org/whl/cu118
    }
} else {
    Write-Host "LLM 해설 기능이 필요하면: .\setup.ps1 -Gpu  (NVIDIA GPU 필요)"
}
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
