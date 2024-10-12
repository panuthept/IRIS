from typing import Optional
from iris.model_wrappers.guard_models.base import GuardLLM
from iris.model_wrappers.guard_models.wild_guard import WildGuard
from iris.model_wrappers.guard_models.llama_guard import LlamaGuard
from iris.model_wrappers.guard_models.shield_gemma import ShieldGemma
from iris.model_wrappers.guard_models.cfi_guard import CFIGuard, DummyBiasModel


AVAILABLE_GUARDS = {
    "WildGuard": WildGuard,
    "LlamaGuard": LlamaGuard,
    "ShieldGemma": ShieldGemma,
}


def load_guard_model(
    guard_name: str, 
    model_name: str, 
    checkpoint_path: Optional[str] = None, 
    api_key: Optional[str] = None, 
    api_base: Optional[str] = None,
) -> GuardLLM:
    assert guard_name in AVAILABLE_GUARDS, f"Invalid guard model: {guard_name}"
    return AVAILABLE_GUARDS[guard_name](
        model_name_or_path=model_name,
        checkpoint_path=checkpoint_path,
        api_key=api_key,
        api_base=api_base,
    )