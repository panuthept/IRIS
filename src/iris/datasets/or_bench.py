from typing import List, Dict
from datasets import load_dataset
from collections import defaultdict
from iris.datasets.base import JailbreakDataset


class ORBenchDataset(JailbreakDataset):
    def __init__(
            self, 
            category: str = None,
            intention: str = None,
            attack_engine: str = None,
            path: str = "./data/datasets/or_bench",
            cache_dir: str = "./data/datasets/or_bench",
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
        return ["test"]
    
    @classmethod
    def intentions_available(cls) -> List[str]:
        return ["harmful", "benign", "hard_benign"]
    
    def _load_dataset(self, path: str) -> Dict[str, List]:
        data: Dict[str, List] = defaultdict(list)
        # Load dataset
        if self.intention is None or self.intention == "harmful":
            dataset = load_dataset("bench-llm/or-bench", "or-bench-toxic", cache_dir=self.cache_dir)
            for sample in dataset["train"]:
                data["test"].append({
                    "instructions": [sample["prompt"]],
                    "instructions_true_label": ["Harmful"],
                })
        if self.intention is None or self.intention == "benign":
            dataset = load_dataset("bench-llm/or-bench", "or-bench-80k", cache_dir=self.cache_dir)
            for sample in dataset["train"]:
                data["test"].append({
                    "instructions": [sample["prompt"]],
                    "instructions_true_label": ["Benign"],
                })
        if self.intention is None or self.intention == "hard_benign":
            dataset = load_dataset("bench-llm/or-bench", "or-bench-hard-1k", cache_dir=self.cache_dir)
            for sample in dataset["train"]:
                data["test"].append({
                    "instructions": [sample["prompt"]],
                    "instructions_true_label": ["Benign"],
                })
        return data
    

if __name__ == "__main__":
    # Use the dataset
    dataset = ORBenchDataset(intention="harmful")
    samples = dataset.as_samples(split="test")
    print(len(samples))
    print(samples[0].get_prompts()[0])
    print(samples[0].instructions_true_label[0])
    print("-" * 100)

    dataset = ORBenchDataset(intention="benign")
    samples = dataset.as_samples(split="test")
    print(len(samples))
    print(samples[0].get_prompts()[0])
    print(samples[0].instructions_true_label[0])
    print("-" * 100)

    dataset = ORBenchDataset(intention="hard_benign")
    samples = dataset.as_samples(split="test")
    print(len(samples))
    print(samples[0].get_prompts()[0])
    print(samples[0].instructions_true_label[0])
    print("-" * 100)