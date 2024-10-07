import json
from typing import List, Dict
from datasets import load_dataset
from collections import defaultdict
from iris.datasets.base import JailbreakDataset


class JailbreakBenchDataset(JailbreakDataset):
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
            category: str = None,
            intention: str = None,
            attack_engine: str = None,
            path: str = "./data/datasets/jailbreak_bench/attack-artifacts",
            cache_dir: str = "./data/datasets/jailbreak_bench/attack-artifacts",
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
    def attack_engines_available(cls) -> List[str]:
        return ["GCG", "JBC", "PAIR", "prompt_with_random_search"]
    
    def _load_dataset(self, path: str) -> Dict[str, List]:
        # Load dataset
        dataset = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", cache_dir=self.cache_dir)
        # Read dataset (original prompt and reference answer)
        test_data: Dict[str, Dict] = defaultdict(dict)
        for sample in dataset["benign"]:
            test_data["benign"][sample["Index"]] = {
                "instructions": [sample["Goal"]],
                "reference_answers": [sample["Target"]],
                "instructions_true_label": ["Benign"],
            }
        for sample in dataset["harmful"]:
            test_data["harmful"][sample["Index"]] = {
                "instructions": [] if self.attack_engine else [sample["Goal"]],
                "reference_instruction": sample["Goal"] if self.attack_engine else None,
                "reference_answers": [sample["Target"]],
                "instructions_true_label": ["Harmful"],
            }
        # Get jailbreak prompts
        if self.attack_engine:
            for instruction_path in self.instruction_paths:
                if instruction_path.startswith(self.attack_engine):
                    with open(f"{path}/{instruction_path}", "r") as f:
                        loaded_data = json.load(f)
                        for sample in loaded_data["jailbreaks"]:
                            if sample["prompt"] is not None:
                                if sample["prompt"] not in test_data["harmful"][sample["index"]]["instructions"]:
                                    test_data["harmful"][sample["index"]]["instructions"].append(sample["prompt"])
                                    test_data["harmful"][sample["index"]]["instructions_true_label"].append("Harmful")
        # Formalize the data
        test_data = {
            "harmful": [sample for sample in test_data["harmful"].values() if len(sample["instructions"]) > 0],
            "benign": [sample for sample in test_data["benign"].values() if len(sample["instructions"]) > 0],
        }
        if self.intention:
            return {"test": test_data[self.intention]}
        else:
            return {"test": test_data["harmful"] + test_data["benign"]}
    

class JailbreakBenchResponseCLFDataset(JailbreakDataset):
    def __init__(
            self, 
            category: str = None,
            intention: str = None,
            attack_engine: str = None,
            path: str = "./data/datasets/jailbreak_bench/attack-artifacts",
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
        return ["harmful"]

    @classmethod
    def attack_engines_available(cls) -> List[str]:
        return ["jailbreak"]
    
    def _load_dataset(self, path: str) -> Dict[str, List]:
        # Load dataset
        dataset = load_dataset("JailbreakBench/JBB-Behaviors", "judge_comparison", cache_dir=self.cache_dir)
        # Read dataset
        test_data: Dict[str, List] = defaultdict(list)
        for sample in dataset["test"]:
            test_data["harmful"].append({
                "instructions": [sample["goal"] if self.attack_engine is None else sample["prompt"]],
                "answers": [sample["target_response"]],
                "reference_instruction": sample["goal"],
                "answers_true_label": ["Harmful" if sample["human_majority"] == 1 else "Benign"]
            })
        return {"test": test_data["harmful"]}


if __name__ == "__main__":
    dataset = JailbreakBenchDataset(intention="harmful")
    samples = dataset.as_samples()
    print("JailbreakBenchDataset:")
    print(f"{samples[0].get_prompts()[0]}\n{samples[0].reference_answers[0]}")
    print("-" * 100)
    dataset = JailbreakBenchDataset(intention="benign")
    samples = dataset.as_samples()
    print(f"{samples[0].get_prompts()[0]}\n{samples[0].reference_answers[0]}")
    print("-" * 100)
    print("=" * 100)

    dataset = JailbreakBenchResponseCLFDataset()
    responses = dataset.as_responses()
    print("JailbreakBenchResponseCLFDataset:")
    print(f"{responses[0].get_prompts()[0]}\n{responses[0].reference_answers[0]}")
    print("-" * 100)
    print(f"{responses[1].get_prompts()[0]}\n{responses[1].reference_answers[0]}")