from typing import List, Dict
from datasets import load_dataset
from collections import defaultdict
from iris.datasets.base import JailbreakDataset


class WildChatDataset(JailbreakDataset):
    def __init__(
            self, 
            category: str = None,
            intention: str = None,
            attack_engine: str = None,
            path: str = "./data/datasets/wildchat",
            cache_dir: str = "./data/datasets/wildchat",
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
    def intentions_available(cls) -> List[str]:
        return ["benign", "harmful"]
    
    def _load_dataset(self, path: str) -> Dict[str, List]:
        data: Dict[str, List] = defaultdict(list)
        # Load train dataset
        dataset = load_dataset("allenai/WildChat-1M-Full", cache_dir=self.cache_dir)
        dataset = dataset.shuffle(seed=42)
        # Read train dataset
        for sample in dataset["train"]:
            if sample["language"] != "English":
                continue
            if sample["turn"] > 1:
                continue
            intention = "harmful" if sample["conversation"][0]["toxic"] else "benign"
            if len(data["train"]) < 365425:
                data["train"].append({
                    "instructions": [sample["conversation"][0]["content"]],
                    "instructions_true_label": [intention.capitalize()],
                })
            else:
                data["test"].append({
                    "instructions": [sample["conversation"][0]["content"]],
                    "instructions_true_label": [intention.capitalize()],
                })
        return data


if __name__ == "__main__":
    dataset = WildChatDataset()
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

    # dataset = load_dataset("allenai/WildChat-1M-Full")
    # print(dataset)
    # print(dataset["train"][0]["conversation"])