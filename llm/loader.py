# llm/loader.py
"""Local LLM loader (optional dependency).

torch / transformers / accelerate / bitsandbytes are imported lazily so that
importing this module (and llm.generate) never fails on a machine without the
GPU stack. Call check_llm_available() to get a clear reason before doing work,
or catch LLMUnavailableError from get_llm().
"""
import importlib.util

# ✅ 기본값 4B로 고정
_DEFAULT_MODEL = "Qwen/Qwen3-4B"

_REQUIRED_MODULES = ("torch", "transformers", "accelerate", "bitsandbytes")
_INSTALL_HINT = (
    "LLM 기능에는 NVIDIA GPU와 requirements-gpu.txt 패키지가 필요합니다: "
    "pip install -r requirements-gpu.txt --extra-index-url https://download.pytorch.org/whl/cu118"
)

_tokenizer = None
_model = None


class LLMUnavailableError(RuntimeError):
    """Local LLM cannot be used on this machine (missing packages or no CUDA GPU)."""


def check_llm_available() -> None:
    """Raise LLMUnavailableError with a human-readable reason if the LLM cannot run."""
    missing = [m for m in _REQUIRED_MODULES if importlib.util.find_spec(m) is None]
    if missing:
        raise LLMUnavailableError(f"LLM 패키지 없음: {', '.join(missing)}. {_INSTALL_HINT}")

    import torch

    if not torch.cuda.is_available():
        raise LLMUnavailableError(
            "CUDA GPU를 찾을 수 없습니다 (4bit 양자화 모델은 GPU 전용). " + _INSTALL_HINT
        )


def get_llm(model_name: str = _DEFAULT_MODEL):
    global _tokenizer, _model
    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model

    check_llm_available()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": 0},  # ✅ GPU-only
        trust_remote_code=True,
    )
    # assign only after both loaded, so a failed load is retried cleanly
    _tokenizer, _model = tokenizer, model
    return _tokenizer, _model
