from typing import List, Dict
from datasets import load_dataset
from collections import defaultdict
from iris.datasets.base import JailbreakDataset


class BeaverTails330kDataset(JailbreakDataset):
    def __init__(
            self, 
            category: str = None,
            intention: str = None,
            attack_engine: str = None,
            path: str = "./data/datasets/beavertails",
            cache_dir: str = "./data/datasets/beavertails",
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
    
    def _load_dataset(self, path: str) -> Dict[str, List]:
        data: Dict[str, List] = defaultdict(list)
        # Load train dataset
        dataset = load_dataset("PKU-Alignment/BeaverTails", cache_dir=self.cache_dir)
        # Read train dataset
        for sample in dataset["330k_train"]:
            intention = "benign" if sample["is_safe"] else "harmful"
            if self.intention is not None and intention != self.intention:
                continue

            data["train"].append({
                "instructions": [sample["prompt"]],
                "instructions_true_label": [intention.capitalize()],
            })

        # Read test dataset
        for sample in dataset["330k_test"]:
            intention = "benign" if sample["is_safe"] else "harmful"
            if self.intention is not None and intention != self.intention:
                continue

            data["test"].append({
                "instructions": [sample["prompt"]],
                "instructions_true_label": [intention.capitalize()],
            })
        return data
    

class BeaverTails30kDataset(BeaverTails330kDataset):
    def _load_dataset(self, path: str) -> Dict[str, List]:
        data: Dict[str, List] = defaultdict(list)
        # Load train dataset
        dataset = load_dataset("PKU-Alignment/BeaverTails", cache_dir=self.cache_dir)
        # Read train dataset
        for sample in dataset["30k_train"]:
            intention = "benign" if sample["is_safe"] else "harmful"
            if self.intention is not None and intention != self.intention:
                continue

            data["train"].append({
                "instructions": [sample["prompt"]],
                "instructions_true_label": [intention.capitalize()],
            })

        # Read test dataset
        for sample in dataset["30k_test"]:
            intention = "benign" if sample["is_safe"] else "harmful"
            if self.intention is not None and intention != self.intention:
                continue

            data["test"].append({
                "instructions": [sample["prompt"]],
                "instructions_true_label": [intention.capitalize()],
            })
        return data


if __name__ == "__main__":
    dataset = BeaverTails330kDataset(
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

    dataset = BeaverTails30kDataset(
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