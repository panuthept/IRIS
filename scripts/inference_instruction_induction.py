import os
import torch
import argparse
from typing import List
from iris.utilities.loaders import save_responses
from iris.data_types import Sample, ModelResponse
from llama_index.llms.openai_like import OpenAILike
from iris.datasets import InstructionIndutionDataset
from iris.model_wrappers.generative_models.api_model import APIGenerativeLLM
from iris.model_wrappers.generative_models.huggingface_model import HuggfaceGenerativeLLM

os.environ["TOKENIZERS_PARALLELISM"] = "false"

system_prompt = (
    "Output only 'Positive' or 'Negative' without additional information."
)
post_processing = lambda x: "positive" if x.strip().capitalize() == "Positive" else "negative"


if __name__ == "__main__":
    parser = parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--api_base", type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=512)
    args = parser.parse_args()

    dataset = InstructionIndutionDataset(
        task_name=args.task_name, 
    )
    samples: List[Sample] = dataset.as_samples()

    if args.api_key:
        model =  APIGenerativeLLM(
            llm=OpenAILike(
                model=args.model_name,
                api_key=args.api_key,
                api_base=args.api_base,
            ),
            system_prompt=system_prompt,
            post_processing=post_processing,
            cache_path="./cache",
        )
    else:
        print(f"CUDA available: {torch.cuda.is_available()}")
        model = HuggfaceGenerativeLLM(
            args.model_name,
            max_tokens=args.max_tokens,
            pipeline_kwargs={
                "torch_dtype": torch.bfloat16,
                "model_kwargs": {
                    "cache_dir": "./data/models",
                    "local_files_only": False,
                }
            },
            system_prompt=system_prompt,
            post_processing=post_processing,
            cache_path="./cache",
        )
        print(f"Device: {model.llm.device}")
    responses: List[ModelResponse] = model.complete_batch(samples)

    # Save the responses to a file
    output_dir = f"./outputs/InstructionIndutionDataset/{args.task_name}/{args.model_name}"
    os.makedirs(output_dir, exist_ok=True)
    save_responses(responses, f"{output_dir}/response.jsonl")