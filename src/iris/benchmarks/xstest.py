from typing import List, Tuple
from iris.datasets import XSTestDataset
from iris.prompt_template import PromptTemplate
from iris.benchmarks.base import JailbreakBenchmark, JailbreakPromptCLFBenchmark


class XSTestBenchmark(JailbreakBenchmark):
    def __init__(
        self, 
        prompt_template: PromptTemplate = None,
        save_path: str = f"./outputs/XSTestBenchmark",
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
        ]

    def get_dataset(self, intention: str, category: str, attack_engine: str) -> XSTestDataset:
        return XSTestDataset(
            intention=intention,
            category=category,
            attack_engine=attack_engine,
        )
    

class XSTestPromptCLFBenchmark(JailbreakPromptCLFBenchmark):
    def __init__(
        self, 
        prompt_template: PromptTemplate = None,
        save_path: str = f"./outputs/XSTestPromptCLFBenchmark",
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
        ]

    def get_dataset(self, intention: str, category: str, attack_engine: str) -> XSTestDataset:
        return XSTestDataset(
            intention=intention,
            category=category,
            attack_engine=attack_engine,
        )


if __name__ == "__main__":
    import torch
    from iris.model_wrappers.generative_models import HuggfaceGenerativeLLM

    benchmark = XSTestBenchmark()

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