import pandas as pd
from typing import List, Dict, Set
from collections import defaultdict
from iris.datasets.base import JailbreakDataset


class JailbreaKV28kDataset(JailbreakDataset):
    def __init__(
            self, 
            category: str = None,
            intention: str = None,
            attack_engine: str = None,
            path: str = "./data/datasets/jailbreakv_28k/JailBreakV_28K",
    ):
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
    def intentions_available(cls) -> List[str]:
        return ["harmful"]

    @classmethod
    def categories_available(cls) -> List[str]:
        return [
            "illegal_activity", 
            "violence",
            "hate_speech",
            "malware",
            "physical_harm",
            "economic_harm",
            "fraud",
            "child_abuse",
            "animal_abuse",
            "political_sensitivity",
            "privacy_violation",
            "tailored_unlicensed_advice",
            "health_consultation",
            "government_decision",
            "unethical_behavior",
            "bias",
        ]

    @classmethod
    def attack_engines_available(cls) -> List[str]:
        return ["template", "persuade", "logic"]

    def _load_dataset(self, path: str) -> Dict[str, List]:
        # Load dataset
        dataset = pd.read_csv(f"{path}/JailBreakV_28K.csv")
        # Read dataset
        test_data: Dict[str, Set] = defaultdict(set)
        for idx in range(len(dataset)):
            sample = dataset.iloc[idx]
            category = sample["policy"].replace(" ", "_").lower()
            attack_engine = sample["format"].lower()

            if category not in self.categories_available():
                continue
            if attack_engine not in self.attack_engines_available():
                continue
            if self.category and category != self.category:
                continue
            if self.attack_engine and attack_engine != self.attack_engine:
                continue

            test_data[sample["redteam_query"]].add(sample["jailbreak_query"])
        # Formalize the data
        test_data = [{
            "instructions": list(jailbreak_queries) if self.attack_engine else [redteam_query],
            "reference_instruction": redteam_query if self.attack_engine else None,
            "instructions_true_label": ["Harmful"] * len(jailbreak_queries) if self.attack_engine else ["Harmful"],
        } for redteam_query, jailbreak_queries in test_data.items()]
        return {"test": [sample for sample in test_data if len(sample["instructions"]) > 0]}


if __name__ == "__main__":
    dataset = JailbreaKV28kDataset()
    print(dataset.get_size())
    samples = dataset.as_samples()
    print(f"{samples[0].get_prompts()[0]}")