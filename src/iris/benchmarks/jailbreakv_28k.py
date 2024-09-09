from typing import List, Tuple
from iris.datasets import JailbreaKV28kDataset
from iris.prompt_template import PromptTemplate
from iris.benchmarks.base import JailbreakBenchmark


class JailbreaKV28kBenchmark(JailbreakBenchmark):
    def __init__(
        self, 
        prompt_template: PromptTemplate = None,
        save_path: str = f"./outputs/JailbreaKV28kBenchmark",
    ):
        super().__init__(
            prompt_template=prompt_template, 
            save_path=save_path
        )

    def get_evaluation_settings(self) -> List[Tuple[str, str, str, str, str]]:
        # Return a list of [{"intention": string, "category": string, "attack_engine": string, "save_name": string, "setting_name": string}, ...]
        return [
            {"intention": "harmful", "save_name": "harmful", "setting_name": "Harmful (Original)"},
            {"intention": "harmful", "attack_engine": "template", "save_name": "harmful_template", "setting_name": "Harmful (Template)"},
            {"intention": "harmful", "attack_engine": "persuade", "save_name": "harmful_persuade", "setting_name": "Harmful (Persuade)"},
            {"intention": "harmful", "attack_engine": "logic", "save_name": "harmful_logic", "setting_name": "Harmful (Logic)"},
        ]

    def get_dataset(self, intention: str, category: str, attack_engine: str) -> JailbreaKV28kDataset:
        return JailbreaKV28kDataset(
            intention=intention,
            category=category,
            attack_engine=attack_engine,
        )


if __name__ == "__main__":
    import torch
    from iris.model_wrappers.generative_models import HuggfaceGenerativeLLM

    benchmark = JailbreaKV28kBenchmark()

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