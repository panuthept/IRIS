from typing import List, Dict
from datasets import load_dataset
from collections import defaultdict
from iris.datasets.base import JailbreakDataset


class OpenAIModerationDataset(JailbreakDataset):
    def __init__(
            self, 
            category: str = None,
            intention: str = None,
            attack_engine: str = None,
            path: str = "./data/datasets/openai_moderation",
            cache_dir: str = "./data/datasets/openai_moderation",
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
        return ["benign", "harmful"]
    
    def _load_dataset(self, path: str) -> Dict[str, List]:
        data: Dict[str, List] = defaultdict(list)
        # Load train dataset
        dataset = load_dataset("mmathys/openai-moderation-api-evaluation", cache_dir=self.cache_dir)
        dataset = dataset.shuffle(seed=42)
        # Read train dataset
        for sample in dataset["train"]:
            content = sample["prompt"]
            intention = "benign"
            for label in ["S", "H", "V", "HR", "SH", "S3", "H2", "V2"]:
                if sample[label] is not None and sample[label] == 1:
                    intention = "harmful"
                    break
            data["test"].append({
                "instructions": [content],
                "instructions_true_label": [intention.capitalize()],
            })
        return data


if __name__ == "__main__":
    dataset = OpenAIModerationDataset()
    samples = dataset.as_samples(split="test")
    print(len(samples))
    print(samples[0].get_prompts()[0])
    print(samples[0].instructions_true_label[0])
    print("-" * 100)
    print(samples[1].get_prompts()[0])
    print(samples[1].instructions_true_label[0])
    print("-" * 100)