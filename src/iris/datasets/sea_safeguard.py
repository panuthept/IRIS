import pandas as pd
from typing import List, Dict
from datasets import load_dataset
from collections import defaultdict
from iris.datasets.base import JailbreakDataset


class SEASafeguardDataset(JailbreakDataset):
    def __init__(
            self, 
            category: str = None,
            intention: str = None,
            language: str = "en",
            cultural: str = "th",
            subset: str = "general",
            sensitive_as_harmful: bool = False,
            attack_engine: str = None,
            path: str = "./data/datasets/sea_safeguard",
            cache_dir: str = "./data/datasets/sea_safeguard",
    ):
        self.cache_dir = cache_dir
        assert cultural in self.cultural_available(), f"Invalid cultural: {cultural}"
        self.cultural = cultural
        assert subset in self.subset_available(), f"Invalid subset: {subset}"
        self.subset = subset
        self.sensitive_as_harmful = sensitive_as_harmful
        super().__init__(
            path=path,
            category=category,
            intention=intention,
            language=language,
            attack_engine=attack_engine
        )

    @classmethod
    def split_available(cls) -> List[str]:
        return ["train", "dev", "test"]
    
    @classmethod
    def subset_available(cls) -> List[str]:
        return ["general", "cultural_specific"]
    
    @classmethod
    def cultural_available(cls) -> List[str]:
        return ["ta", "th", "tl", "ms", "in", "my", "vi"]
    
    @classmethod
    def language_available(cls) -> List[str]:
        return ["en", "ta", "th", "tl", "ms", "in", "my", "vi"]
    
    def _load_dataset(self, path: str) -> Dict[str, List]:
        data: Dict[str, List] = defaultdict(list)

        if self.subset == "general":
            # Load train dataset
            train_dataset = load_dataset("aisingapore/SEASafeguardMix", self.language, cache_dir=self.cache_dir)
            for sample in train_dataset["train"]:
                data["train"].append({
                    "prompt": sample["prompt"] if isinstance(sample["response_label"], str) else None,
                    "response": sample["response"] if isinstance(sample["response_label"], str) else None,
                    "prompt_gold_label": sample["prompt_label"] if isinstance(sample["response_label"], str) else None,
                    "response_gold_label": sample["response_label"] if isinstance(sample["response_label"], str) else None,
                })

            # Load dev dataset
            dev_dataset = pd.read_csv(f"{path}/dev.csv")
            # Read dev dataset
            for i in range(len(dev_dataset)):
                sample = dev_dataset.iloc[i]
                data["dev"].append({
                    "prompt": sample[f"{self.language}_prompt"] if isinstance(sample["response_label"], str) else None,
                    "response": sample[f"{self.language}_response"] if isinstance(sample["response_label"], str) else None,
                    "prompt_gold_label": sample["prompt_label"] if isinstance(sample["response_label"], str) else None,
                    "response_gold_label": sample["response_label"] if isinstance(sample["response_label"], str) else None,
                })

            # Load test dataset
            test_dataset = pd.read_csv(f"{path}/test.csv")
            # Read test dataset
            for i in range(len(test_dataset)):
                sample = test_dataset.iloc[i]
                data["test"].append({
                    "prompt": sample[f"{self.language}_prompt"] if isinstance(sample["response_label"], str) else None,
                    "response": sample[f"{self.language}_response"] if isinstance(sample["response_label"], str) else None,
                    "prompt_gold_label": sample["prompt_label"] if isinstance(sample["response_label"], str) else None,
                    "response_gold_label": sample["response_label"] if isinstance(sample["response_label"], str) else None,
                })
        else:
            # Load test dataset
            test_dataset = load_dataset("aisingapore/SEASafeguardMix", f"{self.cultural}_cultural_specific", cache_dir=self.cache_dir)
            # Read test dataset
            for sample in test_dataset["test"]:
                prompt = sample["en_prompt"] if self.language == "en" else sample["local_prompt"]
                response = sample["en_response"] if self.language == "en" else sample["local_response"]
                prompt_gold_label = sample["prompt_label_final"] if sample["prompt_label_final"] != "Sensitive" else "Harmful" if self.sensitive_as_harmful else "Safe"
                response_gold_label = sample["response_label_final"] if sample["response_label_final"] != "Sensitive" else "Harmful" if self.sensitive_as_harmful else "Safe"
                data["test"].append({
                    "prompt": prompt if isinstance(prompt, str) else None,
                    "response": response if isinstance(response, str) else None,
                    "prompt_gold_label": prompt_gold_label if isinstance(prompt, str) else None,
                    "response_gold_label": response_gold_label if isinstance(response, str) else None,
                })
        return data


if __name__ == "__main__":
    dataset = SEASafeguardDataset(
        language="th",
    )
    samples = dataset.as_samples(split="test")
    print(len(samples))
    print(samples[0].prompt)
    print(samples[0].prompt_gold_label)
    print(samples[0].response)
    print(samples[0].response_gold_label)
    print("-" * 100)

    samples = dataset.as_samples(split="dev")
    print(len(samples))
    print(samples[0].prompt)
    print(samples[0].prompt_gold_label)
    print(samples[0].response)
    print(samples[0].response_gold_label)
    print("-" * 100)

    samples = dataset.as_samples(split="train")
    print(len(samples))
    print(samples[0].prompt)
    print(samples[0].prompt_gold_label)
    print(samples[0].response)
    print(samples[0].response_gold_label)
    print("=" * 100)