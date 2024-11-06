from typing import List, Dict
from datasets import load_dataset
from collections import defaultdict
from iris.datasets.base import JailbreakDataset


class SquadDataset(JailbreakDataset):
    def __init__(
            self, 
            category: str = None,
            intention: str = None,
            attack_engine: str = None,
            path: str = "./data/datasets/squad",
            cache_dir: str = "./data/datasets/squad",
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
        return ["benign"]
    
    def _load_dataset(self, path: str) -> Dict[str, List]:
        data: Dict[str, List] = defaultdict(list)
        # Load train dataset
        dataset = load_dataset("rajpurkar/squad", cache_dir=self.cache_dir)
        # Read train dataset
        for sample in dataset["train"]:
            intention = "benign"
            data["train"].append({
                "instructions": [sample["question"]],
                "instructions_true_label": [intention.capitalize()],
            })

        # Read test dataset
        for sample in dataset["validation"]:
            intention = "benign"
            data["test"].append({
                "instructions": [sample["question"]],
                "instructions_true_label": [intention.capitalize()],
            })
        return data


if __name__ == "__main__":
    dataset = SquadDataset()
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