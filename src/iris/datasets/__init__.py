from typing import Optional
from iris.datasets.squad import SquadDataset
from iris.datasets.xstest import XSTestDataset
from iris.datasets.tydiqa import TyDiQADataset
from iris.datasets.or_bench import ORBenchDataset
from iris.datasets.wildchat import WildChatDataset
from iris.datasets.toxic_chat import ToxicChatDataset
from iris.datasets.base import Dataset, JailbreakDataset
from iris.datasets.wildguardmix import WildGuardMixDataset
from iris.datasets.sea_safeguard import SEASafeguardDataset
from iris.datasets.jailbreakv_28k import JailbreaKV28kDataset
from iris.datasets.awesome_prompts import AwesomePromptsDataset
from iris.datasets.jailbreak_bench import JailbreakBenchDataset
from iris.datasets.promptsource_bench import PromptSourceDataset
from iris.datasets.overshadowprompt import OvershadowPromptDataset
from iris.datasets.openai_moderation import OpenAIModerationDataset
from iris.datasets.instruction_induction import InstructionIndutionDataset
from iris.datasets.beavertails import BeaverTails330kDataset, BeaverTails30kDataset
from iris.datasets.aegis_ai_content_safety import AegisAIContentSafetyDataset, AegisAIContentSafety2Dataset


AVAILABLE_DATASETS = {
    "SquadDataset": SquadDataset,
    "XSTestDataset": XSTestDataset,
    "TyDiQADataset": TyDiQADataset,
    "ORBenchDataset": ORBenchDataset,
    "WildChatDataset": WildChatDataset,
    "ToxicChatDataset": ToxicChatDataset,
    "SEASafeguardDataset": SEASafeguardDataset,
    "WildGuardMixDataset": WildGuardMixDataset,
    "JailbreaKV28kDataset": JailbreaKV28kDataset,
    "AwesomePromptsDataset": AwesomePromptsDataset,
    "JailbreakBenchDataset": JailbreakBenchDataset,
    "BeaverTails30kDataset": BeaverTails30kDataset,
    "BeaverTails330kDataset": BeaverTails330kDataset,
    "OpenAIModerationDataset": OpenAIModerationDataset,
    "OvershadowPromptDataset": OvershadowPromptDataset,
    "AegisAIContentSafetyDataset": AegisAIContentSafetyDataset,
    "AegisAIContentSafety2Dataset": AegisAIContentSafety2Dataset,
}


def load_dataset(
    dataset_name: str, 
    intention: Optional[str] = None, 
    attack_engine: Optional[str] = None,
    language: Optional[str] = "en",
    **kwargs,
) -> Dataset:
    assert dataset_name in AVAILABLE_DATASETS, f"Invalid dataset name: {dataset_name}"
    return AVAILABLE_DATASETS[dataset_name](
        intention=intention,
        attack_engine=attack_engine,
        language=language,
        **kwargs,
    )