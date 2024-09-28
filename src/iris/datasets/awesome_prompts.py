import pandas as pd
from typing import List, Dict
from collections import defaultdict
from iris.datasets.base import JailbreakDataset


class AwesomePromptsDataset(JailbreakDataset):
    """
    Download link: https://github.com/f/awesome-chatgpt-prompts
    """
    def __init__(
            self, 
            category: str = None,
            intention: str = None,
            attack_engine: str = None,
            path: str = "./data/datasets/awesome_prompts",
            cache_dir: str = None,
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
        return ["benign"]
    
    def _load_dataset(self, path: str) -> Dict[str, List]:
        data: Dict[str, List] = defaultdict(list)
        # Load test dataset
        test_dataset = pd.read_csv(f"{path}/prompts.csv")
        for row_idx in range(len(test_dataset)):
            data["test"].append({
                "instructions": [test_dataset.iloc[row_idx]["prompt"]],
                "instructions_true_label": ["Benign"],
            })
        return data


if __name__ == "__main__":
    dataset = AwesomePromptsDataset(
        intention="benign",
        cache_dir="./data/datasets/awesome_prompts",
    )
    samples = dataset.as_samples(split="test")
    print(len(samples))
    print(samples[0].get_prompts()[0])
    print("-" * 100)