import pandas as pd
from typing import List, Dict
from collections import defaultdict
from iris.datasets.base import JailbreakDataset


class XSTestDataset(JailbreakDataset):
    def __init__(
            self, 
            category: str = None,
            intention: str = None,
            attack_engine: str = None,
            path: str = "./data/datasets/xstest",
    ):
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
        super().__init__(
            path=path,
            category=category,
            intention=intention,
            attack_engine=attack_engine
        )

    @classmethod
    def split_available(cls) -> List[str]:
        return ["test"]

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
        test_data: Dict[str, List] = defaultdict(list)
        for idx in range(len(dataset)):
            sample = dataset.iloc[idx]
            category = sample["type"]
            is_harmful = category.startswith("contrast_")
            categories = self.category_mapper[category] if is_harmful else [category]

            if self.category and self.category not in categories:
                continue

            if is_harmful:
                test_data["benign"].append({
                    "instructions": [sample["prompt"]],
                    "instructions_true_label": ["Harmful"],
                })
            else:
                test_data["harmful"].append({
                    "instructions": [sample["prompt"]],
                    "instructions_true_label": ["Benign"],
                })
        # Formalize the data
        if self.intention:
            return {"test": test_data[self.intention]}
        else:
            return {"test": test_data["harmful"] + test_data["benign"]}


if __name__ == "__main__":
    dataset = XSTestDataset(intention="harmful")
    print(dataset.get_size())
    samples = dataset.as_samples()
    print(f"{samples[0].get_prompts()[0]}")

    dataset = XSTestDataset(intention="benign")
    print(dataset.get_size())
    samples = dataset.as_samples()
    print(f"{samples[0].get_prompts()[0]}")