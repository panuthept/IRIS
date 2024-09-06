from typing import List, Tuple
from iris.datasets import JailbreakBenchDataset
from iris.prompt_template import PromptTemplate
from iris.benchmarks.base import JailbreakBenchmark


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
        # Return a list of [(intention, category, attack_engine, save_name, setting_name), ...]
        return [
            ("benign", None, None, "benign", "Benign (Original)"),
            ("harmful", None, None, "harmful", "Harmful (Original)"),
            ("harmful", None, "GCG", "harmful_gcg", "Harmful (GCG)"),
            ("harmful", None, "JBC", "harmful_jbc", "Harmful (JBC)"),
            ("harmful", None, "PAIR", "harmful_pair", "Harmful (PAIR)"),
            ("harmful", None, "prompt_with_random_search", "harmful_prompt_with_random_search", "Harmful (Prompt with Random Search)"),
        ]

    def get_dataset(self, intention: str, category: str, attack_engine: str) -> JailbreakBenchDataset:
        return JailbreakBenchDataset(
            intention=intention,
            category=category,
            attack_engine=attack_engine,
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