from typing import List, Dict
from datasets import load_dataset
from collections import defaultdict
from iris.datasets.base import JailbreakDataset


class ToxicChatDataset(JailbreakDataset):
    def __init__(
            self, 
            category: str = None,
            intention: str = None,
            attack_engine: str = None,
            path: str = "./data/datasets/toxic_chat",
            cache_dir: str = "./data/datasets/toxic_chat",
    ):
        self.cache_dir = cache_dir
        super().__init__(
            path=path,
            category=category,
            intention=intention,
            attack_engine=attack_engine
        )

    @classmethod
    def split_available(cls) -> List[str]:
        return ["train", "test"]

    @classmethod
    def attack_engines_available(cls) -> List[str]:
        return ["vanilla", "jailbreaking"]
    
    def _load_dataset(self, path: str) -> Dict[str, List]:
        data: Dict[str, List] = defaultdict(list)
        # Load train dataset
        dataset = load_dataset("lmsys/toxic-chat", "toxicchat0124", cache_dir=self.cache_dir)
        # Read train dataset
        for sample in dataset["train"]:
            if not sample["human_annotation"]:
                continue
            if self.attack_engine is not None:
                if self.attack_engine == "vanilla" and sample["jailbreaking"]:
                    continue
                if self.attack_engine == "jailbreaking" and not sample["jailbreaking"]:
                    continue

            intention = "benign" if sample["toxicity"] == 0 else "harmful"
            if self.intention is not None and intention != self.intention:
                continue

            data["train"].append({
                "instructions": [sample["user_input"]],
                "instructions_true_label": [intention.capitalize()],
            })

        # Read test dataset
        for sample in dataset["test"]:
            if not sample["human_annotation"]:
                continue
            if self.attack_engine is not None:
                if self.attack_engine == "vanilla" and sample["jailbreaking"]:
                    continue
                if self.attack_engine == "jailbreaking" and not sample["jailbreaking"]:
                    continue

            intention = "benign" if sample["toxicity"] == 0 else "harmful"
            if self.intention is not None and intention != self.intention:
                continue

            data["test"].append({
                "instructions": [sample["user_input"]],
                "instructions_true_label": [intention.capitalize()],
            })
        return data


if __name__ == "__main__":
    dataset = ToxicChatDataset(
        intention="harmful",
    )
    samples = dataset.as_samples(split="test")
    print(len(samples))
    print(samples[0].get_prompts()[0])
    print(samples[0].instructions_true_label[0])
    print("-" * 100)

    samples = dataset.as_samples(split="train")
    print(len(samples))
    print(samples[0].get_prompts()[0])
    print(samples[0].instructions_true_label[0])
    print("=" * 100)