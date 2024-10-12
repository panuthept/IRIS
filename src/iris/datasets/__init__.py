from typing import Optional
from iris.datasets.xstest import XSTestDataset
from iris.datasets.base import Dataset, JailbreakDataset
from iris.datasets.wildguardmix import WildGuardMixDataset
from iris.datasets.jailbreakv_28k import JailbreaKV28kDataset
from iris.datasets.awesome_prompts import AwesomePromptsDataset
from iris.datasets.jailbreak_bench import JailbreakBenchDataset
from iris.datasets.promptsource_bench import PromptSourceDataset
from iris.datasets.instruction_induction import InstructionIndutionDataset


AVAILABLE_DATASETS = {
    "XSTestDataset": XSTestDataset,
    "WildGuardMixDataset": WildGuardMixDataset,
    "JailbreaKV28kDataset": JailbreaKV28kDataset,
    "AwesomePromptsDataset": AwesomePromptsDataset,
    "JailbreakBenchDataset": JailbreakBenchDataset,
}


def load_dataset(
    dataset_name: str, 
    intention: Optional[str] = None, 
) -> Dataset:
    assert dataset_name in AVAILABLE_DATASETS, f"Invalid dataset name: {dataset_name}"
    return AVAILABLE_DATASETS[dataset_name](
        intention=intention,
    )