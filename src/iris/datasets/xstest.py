import pandas as pd
from typing import List, Dict
from iris.data_types import Sample
from collections import defaultdict
from iris.datasets.base import Dataset
from iris.prompt_template import PromptTemplate


class XSTestDataset(Dataset):
    def __init__(
            self, 
            category: str = None,
            attack_engine: str = None,
            path: str = "./data/datasets/xstest",
    ):
        if category:
            assert category in self.categories_available(), f"Category {category} not available"

        self.category_mapper = {
            "contrast_homonyms": ["homonyms"],
            "contrast_figurative_language": ["figurative_language"],
            "contrast_safe_targets": ["safe_targets"],
            "contrast_safe_contexts": ["safe_contexts"],
            "contrast_definitions": ["definitions"],
            "contrast_discr": ["real_group_nons_discr", "nons_group_real_discr"],
            "contrast_historical_events": ["historical_events"],
            "contrast_privacy": ["privacy_public", "privacy_fictional"],
        }

        self.category = category
        self.data = self._load_dataset(path)

    @classmethod
    def split_available(cls) -> List[str]:
        return ["harmful", "benign"]

    @classmethod
    def categories_available(cls) -> List[str]:
        return [
            "homonyms", 
            "figurative_language",
            "safe_targets",
            "safe_contexts",
            "definitions",
            "real_group_nons_discr",
            "nons_group_real_discr",
            "historical_events",
            "privacy_public",
            "privacy_fictional",
        ]

    def _load_dataset(self, path: str) -> Dict[str, List]:
        # Load dataset
        dataset = pd.read_csv(f"{path}/xstest_v2_prompts.csv")
        # Read dataset
        samples: Dict[str, List] = defaultdict(list)
        for idx in range(len(dataset)):
            sample = dataset.iloc[idx]
            category = sample["type"]
            is_benign = category.startswith("contrast_")
            categories = self.category_mapper[category] if is_benign else [category]

            if self.category and self.category not in categories:
                continue

            if is_benign:
                samples["benign"].append({"instructions": [sample["prompt"]]})
            else:
                samples["harmful"].append({"instructions": [sample["prompt"]]})
        return samples

    def get_size(self, split="harmful") -> int:
        assert split in self.split_available(), f"Split {split} not available"
        return len(self.data[split])

    def as_samples(self, split="harmful", prompt_template: PromptTemplate = None) -> List[Sample]:
        assert split in self.split_available(), f"Split {split} not available"

        samples: List[Sample] = []
        for sample in self.data[split]:
            samples.append(
                Sample(
                    instructions=sample["instructions"],
                    prompt_template=PromptTemplate(
                        instruction_template="{instruction}",
                    ) if prompt_template is None else prompt_template
                )
            )
        return samples


if __name__ == "__main__":
    dataset = XSTestDataset()
    print(dataset.get_size(split="harmful"))
    print(dataset.get_size(split="benign"))
    samples = dataset.as_samples(split="harmful")
    print(f"{samples[0].get_prompts()[0]}")
    samples = dataset.as_samples(split="benign")
    print(f"{samples[0].get_prompts()[0]}")