from typing import List, Tuple
from iris.prompt_template import PromptTemplate
from iris.benchmarks.base import JailbreakBenchmark
from iris.datasets import JailbreakBenchDataset, JailbreakBenchPromptCLFDataset


class JailbreakBenchBenchmark(JailbreakBenchmark):
    def __init__(
        self, 
        prompt_template: PromptTemplate = None,
        save_path: str = f"./outputs/JailbreakBenchBenchmark",
    ):
        super().__init__(
            prompt_template=prompt_template, 
            save_path=save_path
        )

    def get_evaluation_settings(self) -> List[Tuple[str, str, str, str, str]]:
        # Return a list of [{"intention": string, "category": string, "attack_engine": string, "save_name": string, "setting_name": string}, ...]
        return [
            {"intention": "benign", "save_name": "benign", "setting_name": "Benign (Original)"},
            {"intention": "harmful", "save_name": "harmful", "setting_name": "Harmful (Original)"},
            {"intention": "harmful", "attack_engine": "GCG", "save_name": "harmful_gcg", "setting_name": "Harmful (GCG)"},
            {"intention": "harmful", "attack_engine": "JBC", "save_name": "harmful_jbc", "setting_name": "Harmful (JBC)"},
            {"intention": "harmful", "attack_engine": "PAIR", "save_name": "harmful_pair", "setting_name": "Harmful (PAIR)"},
            {"intention": "harmful", "attack_engine": "prompt_with_random_search", "save_name": "harmful_prompt_with_random_search", "setting_name": "Harmful (Prompt with Random Search)"},
        ]

    def get_dataset(self, intention: str, category: str, attack_engine: str) -> JailbreakBenchDataset:
        return JailbreakBenchDataset(
            intention=intention,
            category=category,
            attack_engine=attack_engine,
            cache_dir="./data/datasets/jailbreak_bench",
        )
    

class JailbreakBenchPromptCLFBenchmark(JailbreakBenchmark):
    def __init__(
        self, 
        prompt_template: PromptTemplate = None,
        save_path: str = f"./outputs/JailbreakBenchPromptCLFBenchmark",
    ):
        super().__init__(
            prompt_template=prompt_template, 
            save_path=save_path
        )

    def get_dataset(self, intention: str, category: str, attack_engine: str) -> JailbreakBenchPromptCLFDataset:
        return JailbreakBenchPromptCLFDataset(
            intention=intention,
            category=category,
            attack_engine=attack_engine,
            cache_dir="./data/datasets/jailbreak_bench",
        )


if __name__ == "__main__":
    import torch
    from iris.model_wrappers.generative_models import HuggfaceGenerativeLLM

    benchmark = JailbreakBenchBenchmark()

    print(f"CUDA available: {torch.cuda.is_available()}")
    model = HuggfaceGenerativeLLM(
        "Qwen/Qwen2-0.5B-Instruct",
        max_tokens=512,
        pipeline_kwargs={
            "torch_dtype": torch.bfloat16,
            "model_kwargs": {
                "cache_dir": "./data/models",
                "local_files_only": False,
            }
        },
        cache_path="./cache",
    )
    print(f"Device: {model.llm.device}")

    results = benchmark.evaluate(model)
    print(results)