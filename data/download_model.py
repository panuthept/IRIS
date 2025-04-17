import torch
import argparse
from iris.model_wrappers.generative_models.huggingface_model import HuggfaceGenerativeLLM


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-0.5B-Instruct")
    args = parser.parse_args()

    model = HuggfaceGenerativeLLM(
        args.model_name,
    )
    print(model)
    print(f"Downloaded model: {model} successfully!")