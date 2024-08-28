import torch
import transformers


class HuggfaceGenerativeLLM:
    def __init__(self, huggingface_model_name_or_path: str, system_prompt: str = None):
        self.llm = transformers.pipeline(
            "text-generation", 
            model=huggingface_model_name_or_path,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        self.model_name = huggingface_model_name_or_path
        self.system_prompt = system_prompt

    def _complete(self, instruction: str) -> str:
        if self.system_prompt:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": instruction},
            ]
        else:
            messages = [{"role": "user", "content": instruction}]
        answer = self.llm(messages)[0]["generated_text"][-1]
        return answer