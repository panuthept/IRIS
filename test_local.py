import torch
from transformers import AutoModel
from iris.model_wrappers.generative_models.huggingface_model import HuggfaceGenerativeLLM


if __name__ == "__main__":
    model = HuggfaceGenerativeLLM(
        "Qwen/Qwen2-0.5B-Instruct",
        model_kwargs={
            "cache_dir": "./models",
            "torch_dtype": torch.bfloat16,
            "local_files_only": True,
        },
    )
    print(model)

    # model = AutoModel.from_pretrained()