import json
from typing import List, Dict
from datasets import load_dataset
from iris.data_types import Sample
from collections import defaultdict
from iris.datasets.base import Dataset


class JailbreakBenchDataset(Dataset):
    instruction_paths = [
        "GCG/transfer/gpt-3.5-turbo-1106.json",
        "GCG/transfer/gpt-4-0125-preview.json",
        "GCG/white_box/llama-2-7b-chat-hf.json",
        "GCG/white_box/vicuna-13b-v1.5.json",
        "JBC/manual/gpt-3.5-turbo-1106.json",
        "JBC/manual/gpt-4-0125-preview.json",
        "JBC/manual/llama-2-7b-chat-hf.json",
        "JBC/manual/vicuna-13b-v1.5.json",
        "PAIR/black_box/gpt-3.5-turbo-1106.json",
        "PAIR/black_box/gpt-4-0125-preview.json",
        "PAIR/black_box/llama-2-7b-chat-hf.json",
        "PAIR/black_box/vicuna-13b-v1.5.json",
        "prompt_with_random_search/black_box/gpt-3.5-turbo-1106.json",
        "prompt_with_random_search/black_box/gpt-4-0125-preview.json",
        "prompt_with_random_search/black_box/llama-2-7b-chat-hf.json",
        "prompt_with_random_search/black_box/vicuna-13b-v1.5.json",
    ]

    def __init__(
            self, 
            path: str = "./data/datasets/jailbreak_bench/attack-artifacts",
            attack_engine: str = None,
            cache_dir: str = None,
    ):
        if attack_engine:
            assert attack_engine in self.attack_engines_available()
        self.attack_engine = attack_engine
        self.data = self._load_dataset(path, cache_dir=cache_dir)

    def attack_engines_available(self) -> List[str]:
        return ["GCG", "JBC", "PAIR", "prompt_with_random_search"]

    def _load_dataset(self, path: str, cache_dir: str = None) -> Dict[str, List]:
        # Load the behaviors dataset
        dataset = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", cache_dir=cache_dir)
        # Get the original instruction and reference answer
        samples = defaultdict(dict)
        for data in dataset["harmful"]:
            samples["harmful"][data["Index"]] = {
                "instructions": [],
                "reference_answers": [data["Target"]]
            }
            if self.attack_engine is None:
                samples["harmful"][data["Index"]]["instructions"].append(data["Goal"])
        for data in dataset["benign"]:
            samples["benign"][data["Index"]] = {
                "instructions": [data["Goal"]],
                "reference_answers": [data["Target"]]
            }
        # Get the attacking instructions
        if self.attack_engine is not None:
            for instruction_path in self.instruction_paths:
                if instruction_path.startswith(self.attack_engine):
                    with open(f"{path}/{instruction_path}", "r") as f:
                        loaded_data = json.load(f)
                        for data in loaded_data["jailbreaks"]:
                            if data["prompt"] is not None:
                                samples["harmful"][data["index"]]["instructions"].append(data["prompt"])
        # Load the judge_comparison dataset
        dataset = load_dataset("JailbreakBench/JBB-Behaviors", "judge_comparison", cache_dir=cache_dir)
        for data in dataset["test"]:
            samples["judge_comparison"][data["Index"]] = {
                "instructions": [
                    f'{data["goal"]}\nResponse: {data["target_response"]}', 
                    f'{data["prompt"]}\nResponse: {data["target_response"]}', 
                ],
                "reference_answers": ["Harmful" if data["human_majority"] == 1 else "Benign"]
            }
        # Convert to list and remove sample that has no instructions
        samples["harmful"] = [sample for sample in samples["harmful"].values() if len(sample["instructions"]) > 0]
        samples["benign"] = [sample for sample in samples["benign"].values() if len(sample["instructions"]) > 0]
        samples["judge_comparison"] = [sample for sample in samples["judge_comparison"].values() if len(sample["instructions"]) > 0]
        return samples

    def as_samples(self, split="harmful") -> List[Sample]:
        samples: List[Sample] = []
        for sample in self.data[split]:
            samples.append(
                Sample(
                    instructions=sample["instructions"],
                    reference_answers=sample["reference_answers"],
                    prompt_template={
                        "instruction_only": "{instruction}",
                    }
                )
            )
        return samples


class JailbreakBenchPromptCLFDataset(JailbreakBenchDataset):
    def as_samples(self) -> List[Sample]:
        samples: List[Sample] = []
        for sample in self.data["harmful"]:
            samples.append(
                Sample(
                    instructions=sample["instructions"],
                    reference_answers=["Harmful"],
                )
            )
        for sample in self.data["benign"]:
            samples.append(
                Sample(
                    instructions=sample["instructions"],
                    reference_answers=["Benign"],
                )
            )
        return samples
    

class JailbreakBenchResponseCLFDataset(JailbreakBenchDataset):
    def as_samples(self) -> List[Sample]:
        samples: List[Sample] = []
        for sample in self.data["judge_comparison"]:
            samples.append(
                Sample(
                    instructions=sample["instructions"],
                    reference_answers=sample["reference_answers"],
                )
            )
        return samples


if __name__ == "__main__":
    dataset = JailbreakBenchDataset()
    samples = dataset.as_samples("harmful")
    print("JailbreakBenchDataset:")
    print(f"{samples[0].get_prompts()[0]}{samples[0].reference_answers[0]}")
    print("-" * 100)
    samples = dataset.as_samples("benign")
    print(f"{samples[0].get_prompts()[0]}{samples[0].reference_answers[0]}")
    print("-" * 100)
    print("=" * 100)

    dataset = JailbreakBenchPromptCLFDataset()
    samples = dataset.as_samples()
    print("JailbreakBenchPromptCLFDataset:")
    print(f"{samples[0].get_prompts()[0]}{samples[0].reference_answers[0]}")
    print("-" * 100)
    print(f"{samples[-1].get_prompts()[0]}{samples[-1].reference_answers[0]}")
    print("-" * 100)
    print("=" * 100)

    dataset = JailbreakBenchResponseCLFDataset()
    samples = dataset.as_samples()
    print("JailbreakBenchResponseCLFDataset:")
    print(f"{samples[2].get_prompts()[0]}{samples[2].reference_answers[0]}")
    print("-" * 100)
    print(f"{samples[0].get_prompts()[0]}{samples[0].reference_answers[0]}")