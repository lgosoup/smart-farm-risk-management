# llm/loader.py
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# ✅ 기본값 4B로 고정
_DEFAULT_MODEL = "Qwen/Qwen3-4B"

_tokenizer = None
_model = None


def get_llm(model_name: str = _DEFAULT_MODEL):
    global _tokenizer, _model
    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    _tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    _model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": 0},  # ✅ GPU-only
        trust_remote_code=True,
    )
    return _tokenizer, _model
