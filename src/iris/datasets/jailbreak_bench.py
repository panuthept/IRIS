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

    def __init__(self, path: str = "./data/jailbreak_bench/attack-artifacts"):
        self.data = self._load_dataset(path)

    def _load_dataset(self, path: str) -> Dict[str, List]:
        # Load the dataset
        dataset = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
        # Get the original instruction and reference answer
        samples = defaultdict(dict)
        for data in dataset["harmful"]:
            samples["harmful"][data["Index"]] = {
                "instructions": [data["Goal"]],
                "reference_answers": [data["Target"]]
            }
        for data in dataset["benign"]:
            samples["benign"][data["Index"]] = {
                "instructions": [data["Goal"]],
                "reference_answers": [data["Target"]]
            }
        # Get the attacking instructions
        for instruction_path in self.instruction_paths:
            with open(f"{path}/{instruction_path}", "r") as f:
                loaded_data = json.load(f)
                for data in loaded_data["jailbreaks"]:
                    if data["prompt"] is not None:
                        samples["harmful"][data["index"]]["instructions"].append(data["prompt"])
        samples["harmful"] = list(samples["harmful"].values())
        samples["benign"] = list(samples["benign"].values())
        return samples

    def as_samples(self, split="harmful") -> List[Sample]:
        samples: List[Sample] = [
            Sample(
                instructions=sample["instructions"],
                reference_answers=sample["reference_answers"],
            )
            for sample in self.data[split]
        ]
        return samples


if __name__ == "__main__":
    dataset = JailbreakBenchDataset()
    samples = dataset.as_samples("harmful")
    print(samples[0].instructions)
    print(len(samples[0].instructions))