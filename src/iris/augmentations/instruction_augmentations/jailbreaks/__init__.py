from typing import Callable
from iris.model_wrappers.guard_models import GuardLLM
from iris.augmentations.instruction_augmentations.jailbreaks.base import Jailbreaking
from iris.augmentations.instruction_augmentations.jailbreaks.gpt_fuzzer import GPTFuzzerJailbreaking
from iris.augmentations.instruction_augmentations.jailbreaks.wildteaming import WildTeamingJailbreaking
from iris.augmentations.instruction_augmentations.jailbreaks.multilingual_jailbreak import MultiLingualJailbreaking, MultiLingualJailbreakingPlus
from iris.augmentations.instruction_augmentations.jailbreaks.jailbroken_jailbreak import JailbrokenJailbreaking
from iris.augmentations.instruction_augmentations.jailbreaks.deepinception_jailbreak import DeepInceptionJailbreaking
from iris.augmentations.instruction_augmentations.jailbreaks.ica_jailbreak import ICAJailbreaking


AVAILABLE_ATTACKERS = {
    "GPTFuzzerJailbreaking": GPTFuzzerJailbreaking,
    "WildTeamingJailbreaking": WildTeamingJailbreaking,
    "MultiLingualJailbreaking": MultiLingualJailbreaking,
    "MultiLingualJailbreakingPlus": MultiLingualJailbreakingPlus,
    "JailbrokenJailbreaking": JailbrokenJailbreaking,
    "DeepInceptionJailbreaking": DeepInceptionJailbreaking,
    "ICAJailbreaking": ICAJailbreaking,
}



def load_attacker(
    attacker_name: str, 
    target_model: GuardLLM,
    evaluator: Callable,
    max_iteration: int = 10
) -> Jailbreaking:
    assert attacker_name in AVAILABLE_ATTACKERS, f"Invalid attacker name: {attacker_name}"
    return AVAILABLE_ATTACKERS[attacker_name](
        target_model=target_model,
        evaluator=evaluator,
        max_iteration=max_iteration,
    )