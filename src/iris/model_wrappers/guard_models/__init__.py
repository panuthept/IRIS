from typing import Optional
from iris.model_wrappers.guard_models.base import GuardLLM, PromptGuard
from iris.model_wrappers.guard_models.poly_guard import PolyGuard
# from iris.model_wrappers.guard_models.nemo_guard import NemoGuard
# from iris.model_wrappers.guard_models.wild_guard import WildGuard
from iris.model_wrappers.guard_models.shield_gemma import ShieldGemma
from iris.model_wrappers.guard_models.sealion_guard import SealionGuardAPI, SealionGuard, GemmaSealionGuard
from iris.model_wrappers.guard_models.llama_guard import LlamaGuard, LlamaGuard4
# from iris.model_wrappers.guard_models.cfi_guard import CFIGuard, DummyBiasModel


AVAILABLE_GUARDS = {
    # "NemoGuard": NemoGuard,
    # "WildGuard": WildGuard,
    "LlamaGuard": LlamaGuard,
    "LlamaGuard4": LlamaGuard4,
    "ShieldGemma": ShieldGemma,
    "SealionGuardAPI": SealionGuardAPI,
    "SealionGuard": SealionGuard,
    "GemmaSealionGuard": GemmaSealionGuard,
    "PolyGuard": PolyGuard,
}


def load_safeguard(
    safeguard_name: str, 
    model_name: str, 
    checkpoint_path: Optional[str] = None, 
    api_key: Optional[str] = None, 
    api_base: Optional[str] = None,
    disable_logitlens: bool = False,
    top_logprobs: int = 2,
    max_tokens: int = 3000,
) -> GuardLLM:
    assert safeguard_name in AVAILABLE_GUARDS, f"Invalid guard model: {safeguard_name}"
    return AVAILABLE_GUARDS[safeguard_name](
        model_name_or_path=model_name,
        checkpoint_path=checkpoint_path,
        api_key=api_key,
        api_base=api_base,
        disable_logitlens=disable_logitlens,
        top_logprobs=top_logprobs,
        max_tokens=max_tokens,
    )