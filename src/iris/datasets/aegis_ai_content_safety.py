import json
from typing import List, Dict
from datasets import load_dataset
from collections import defaultdict
from iris.datasets.base import JailbreakDataset


class AegisAIContentSafetyDataset(JailbreakDataset):
    def __init__(
            self, 
            category: str = None,
            intention: str = None,
            attack_engine: str = None,
            path: str = "./data/datasets/aegis_ai_content_safety",
            cache_dir: str = "./data/datasets/aegis_ai_content_safety",
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
        return ["harmful", "benign"]

    def _load_dataset(self, path: str) -> Dict[str, List]:
        data: Dict[str, List] = defaultdict(list)
        # Load train dataset
        dataset = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-1.0", cache_dir=self.cache_dir)
        # Read train dataset
        for sample in dataset["train"]:
            voted = 0
            for i in range(0, 5):
                if sample[f"labels_{i}"] is None:
                    continue

                if "Safe" in sample[f"labels_{i}"]:
                    voted += 1
            
            groundtruth = round(sample["num_annotations"]/2)
            intention = "benign" if voted >= groundtruth else "harmful"
            if self.intention is not None and intention != self.intention:
                continue
            data["train"].append({
                "instructions": [sample["text"]],
                "instructions_true_label": [intention.capitalize()],
            })

        for sample in dataset["test"]:
            if sample["labels_0"] is None or sample["labels_1"] is None or sample["labels_2"] is None:
                continue
        
            intention = "benign" if sample["labels_0"] == "Safe" else "harmful"
            if self.intention is not None and intention != self.intention:
                continue
            data["test"].append({
                "instructions": [sample["text"]],
                "instructions_true_label": [intention.capitalize()],
            })
        return data

if __name__ == "__main__":
    dataset = AegisAIContentSafetyDataset(
        intention="harmful",
        cache_dir="./data/datasets/aegis_ai_content_safety",
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