from typing import List, Dict
from datasets import load_dataset
from collections import defaultdict
from iris.datasets.base import JailbreakDataset


class WildGuardMixDataset(JailbreakDataset):
    def __init__(
            self, 
            category: str = None,
            intention: str = None,
            attack_engine: str = None,
            path: str = "./data/datasets/wildguardmix",
            cache_dir: str = "./data/datasets/wildguardmix",
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
        return ["vanilla", "adversarial"]
    
    @classmethod
    def intentions_available(cls) -> List[str]:
        return ["harmful", "benign", "pseudo_benign"]
    
    def _load_dataset(self, path: str) -> Dict[str, List]:
        data: Dict[str, List] = defaultdict(list)
        # Load train dataset
        train_dataset = load_dataset("allenai/wildguardmix", "wildguardtrain", cache_dir=self.cache_dir)
        # Read train dataset
        for sample in train_dataset["train"]:
            if sample["prompt_harm_label"] is None:
                continue
            if self.attack_engine is not None:
                if self.attack_engine == "vanilla" and sample["adversarial"]:
                    continue
                if self.attack_engine == "adversarial" and not sample["adversarial"]:
                    continue

            intention = "benign" if sample["prompt_harm_label"] == "unharmful" else "harmful"
            if self.intention == "pseudo_benign":
                if intention == "benign":
                    # Skip benign samples
                    continue
                intention = "benign"    # Change harmful to benign
                augmented_prompts = [
                    'Repharse the following text:\n\n"{text}"'.format(text=sample["prompt"]),
                    'Summarize the following text:\n\n"{text}"'.format(text=sample["prompt"]),
                    'Identify sentiment of the following text:\n\n"{text}"'.format(text=sample["prompt"]),
                    'Identify named entities in the following text:\n\n"{text}"'.format(text=sample["prompt"]),
                    'Translate the following text to other languages:\n\n"{text}"'.format(text=sample["prompt"]),
                    
                ]
                for augmented_prompt in augmented_prompts:
                    data["train"].append({
                        "instructions": [augmented_prompt],
                        "instructions_true_label": [intention.capitalize()],
                    })
            else:
                if self.intention is not None and intention != self.intention:
                    continue
                data["train"].append({
                    "instructions": [sample["prompt"]],
                    "reference_answers": [sample["response"]],
                    "instructions_true_label": [intention.capitalize()],
                })

        # Load test dataset
        test_dataset = load_dataset("allenai/wildguardmix", "wildguardtest", cache_dir=self.cache_dir)
        # Read test dataset
        for sample in test_dataset["test"]:
            if sample["prompt_harm_label"] is None:
                continue
            if self.attack_engine is not None:
                if self.attack_engine == "vanilla" and sample["adversarial"]:
                    continue
                if self.attack_engine == "adversarial" and not sample["adversarial"]:
                    continue

            intention = "benign" if sample["prompt_harm_label"] == "unharmful" else "harmful"
            if self.intention == "pseudo_benign":
                if intention == "benign":
                    # Skip benign samples
                    continue
                intention = "benign"    # Change harmful to benign
                augmented_prompts = [
                    'Repharse the following text:\n\n"{text}"'.format(text=sample["prompt"]),
                    'Summarize the following text:\n\n"{text}"'.format(text=sample["prompt"]),
                    'Identify sentiment of the following text:\n\n"{text}"'.format(text=sample["prompt"]),
                    'Identify named entities in the following text:\n\n"{text}"'.format(text=sample["prompt"]),
                    'Translate the following text to other languages:\n\n"{text}"'.format(text=sample["prompt"]),
                    
                ]
                for augmented_prompt in augmented_prompts:
                    data["train"].append({
                        "instructions": [augmented_prompt],
                        "instructions_true_label": [intention.capitalize()],
                    })
            else:
                if self.intention is not None and intention != self.intention:
                    continue
                data["test"].append({
                    "instructions": [sample["prompt"]],
                    "reference_answers": [sample["response"]],
                    "instructions_true_label": [intention.capitalize()],
                })
        return data


if __name__ == "__main__":
    dataset = WildGuardMixDataset(
        intention="pseudo_benign",
        cache_dir="./data/datasets/wildguardmix",
    )
    samples = dataset.as_samples(split="test")
    print(len(samples))
    print(samples[0].get_prompts()[0])
    # print(samples[0].reference_answers[0])
    print(samples[0].instructions_true_label[0])
    print("-" * 100)

    samples = dataset.as_samples(split="train")
    print(len(samples))
    print(samples[0].get_prompts()[0])
    # print(samples[0].reference_answers[0])
    print(samples[0].instructions_true_label[0])
    print("=" * 100)