from typing import Optional, Union
from iris.model_wrappers.guard_models.base import GuardLLM, PromptGuard
from iris.model_wrappers.guard_models.poly_guard import PolyGuard
from iris.model_wrappers.guard_models.lakera_guard import LakeraGuard
from iris.model_wrappers.guard_models.seal_guard import SEALGuard
from iris.model_wrappers.guard_models.lion_guard import LionGuard2
from iris.model_wrappers.guard_models.azure_ai_content_safety import AzureAIContentSafety
from iris.model_wrappers.guard_models.google_model_armor import GoogleModelArmor
from iris.model_wrappers.guard_models.openai_moderation import OpenAIModeration
# from iris.model_wrappers.guard_models.nemo_guard import NemoGuard
# from iris.model_wrappers.guard_models.wild_guard import WildGuard
from iris.model_wrappers.guard_models.shield_gemma import ShieldGemma
from iris.model_wrappers.guard_models.llm_guard import LLMGuard, GPT4o, Llama31
from iris.model_wrappers.guard_models.llama_guard import LlamaGuard, LlamaGuard4
from iris.model_wrappers.guard_models.sealion_guard import SealionGuardAPI, SealionGuard, GemmaSealionGuard
# from iris.model_wrappers.guard_models.cfi_guard import CFIGuard, DummyBiasModel


AVAILABLE_GUARDS = {
    # "NemoGuard": NemoGuard,
    # "WildGuard": WildGuard,
    "GPT4o": GPT4o,
    "Llama31": Llama31,
    "LLMGuard": LLMGuard,
    "SEALGuard": SEALGuard,
    "LionGuard2": LionGuard2,
    "OpenAIModeration": OpenAIModeration,
    "LlamaGuard": LlamaGuard,
    "LakeraGuard": LakeraGuard,
    "AzureAIContentSafety": AzureAIContentSafety,
    "GoogleModelArmor": GoogleModelArmor,
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
    **kwargs,
) -> Union[GuardLLM, PromptGuard]:
    assert safeguard_name in AVAILABLE_GUARDS, f"Invalid guard model: {safeguard_name}"
    return AVAILABLE_GUARDS[safeguard_name](
        model_name_or_path=model_name,
        **kwargs,
    )