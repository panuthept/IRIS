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
                if sample["prompt_label"] is None:
                    continue
                intention = "benign" if sample["prompt_label"] == "Safe" else "harmful"
                if self.intention is not None and intention != self.intention:
                    continue
                data["train"].append({
                    "instructions": [sample["prompt"]],
                    "reference_answers": [sample["response"]],
                    "instructions_true_label": [intention.capitalize()],
                })

            # Load dev dataset
            dev_dataset = pd.read_csv(f"{path}/dev.csv")
            # Read dev dataset
            for i in range(len(dev_dataset)):
                sample = dev_dataset.iloc[i]
                if sample["prompt_label"] is None:
                    continue
                intention = "benign" if sample["prompt_label"] == "Safe" else "harmful"
                if self.intention is not None and intention != self.intention:
                    continue
                data["dev"].append({
                    "instructions": [sample[f"{self.language}_prompt"]],
                    "reference_answers": [sample[f"{self.language}_response"]],
                    "instructions_true_label": [intention.capitalize()],
                })

            # Load test dataset
            test_dataset = pd.read_csv(f"{path}/test.csv")
            # Read test dataset
            for i in range(len(test_dataset)):
                sample = test_dataset.iloc[i]
                if sample["prompt_label"] is None:
                    continue
                intention = "benign" if sample["prompt_label"] == "Safe" else "harmful"
                if self.intention is not None and intention != self.intention:
                    continue
                data["test"].append({
                    "instructions": [sample[f"{self.language}_prompt"]],
                    "reference_answers": [sample[f"{self.language}_response"]],
                    "instructions_true_label": [intention.capitalize()],
                })
        else:
            # Load test dataset
            test_dataset = load_dataset("aisingapore/SEASafeguardMix", f"{self.cultural}_cultural_specific", cache_dir=self.cache_dir)
            # Read test dataset
            for sample in test_dataset["test"]:
                if sample["prompt_label_final"] is None:
                    continue
                intention = "benign" if sample["prompt_label_final"] == "Safe" else "harmful" if sample["prompt_label_final"] == "Harmful" else "harmful" if sample["prompt_label_final"] == "Sensitive" and self.sensitive_as_harmful else "benign"
                if self.intention is not None and intention != self.intention:
                    continue
                data["test"].append({
                    "instructions": [sample["en_prompt"] if self.language == "en" else sample["local_prompt"]],
                    "reference_answers": [sample["en_response"]],
                    "instructions_true_label": [intention.capitalize()],
                })
        return data


if __name__ == "__main__":
    dataset = SEASafeguardDataset(
        language="th",
    )
    samples = dataset.as_samples(split="test")
    print(len(samples))
    print(samples[0].get_prompts()[0])
    # print(samples[0].reference_answers[0])
    print(samples[0].instructions_true_label[0])
    print("-" * 100)

    samples = dataset.as_samples(split="dev")
    print(len(samples))
    print(samples[0].get_prompts()[0])
    # print(samples[0].reference_answers[0])
    print(samples[0].instructions_true_label[0])
    print("=" * 100)