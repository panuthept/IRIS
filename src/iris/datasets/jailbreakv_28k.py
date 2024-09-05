import pandas as pd
from typing import List, Dict
from iris.data_types import Sample
from collections import defaultdict
from iris.datasets.base import Dataset
from iris.prompt_template import PromptTemplate


class JailbreaKV28kDataset(Dataset):
    def __init__(
            self, 
            safety_policy: str = None,
            attack_engine: str = None,
            path: str = "./data/datasets/jailbreakv_28k/JailBreakV_28K",
    ):
        if safety_policy:
            assert safety_policy in self.safety_policies_available(), f"Safety policy {safety_policy} not available"
        if attack_engine:
            assert attack_engine in self.attack_engines_available(), f"Attack engine {attack_engine} not available"

        self.safety_policy = safety_policy
        self.attack_engine = attack_engine
        self.data = self._load_dataset(path)

    @classmethod
    def safety_policies_available(cls) -> List[str]:
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

    def _load_dataset(self, path: str, cache_dir: str = None) -> List[Dict]:
        # Load dataset
        dataset = pd.read_csv(f"{path}/JailBreakV_28K.csv")
        # Read dataset
        samples: Dict[str, List] = defaultdict(set)
        for idx in range(len(dataset)):
            sample = dataset.iloc[idx]
            policy = sample["policy"].replace(" ", "_").lower()
            format = sample["format"].lower()

            if policy not in self.safety_policies_available():
                continue
            if format not in self.attack_engines_available():
                continue
            if self.safety_policy and policy != self.safety_policy:
                continue
            if self.attack_engine and format != self.attack_engine:
                continue

            samples[sample["redteam_query"]].add(sample["jailbreak_query"])
        # Convert to list
        data = [{"instructions": list(jailbreak_queries) if self.attack_engine else [redteam_query]} for redteam_query, jailbreak_queries in samples.items()]
        return data

    def get_size(self) -> int:
        return len(self.data)

    def as_samples(self, prompt_template: PromptTemplate = None) -> List[Sample]:
        samples: List[Sample] = []
        for sample in self.data:
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
    dataset = JailbreaKV28kDataset(attack_engine="template", safety_policy="violence")
    print(dataset.get_size())
    samples = dataset.as_samples()
    print(f"{len(samples[0].get_prompts())}")
    print(f"{samples[0].get_prompts()[0]}")