import transformers
from typing import Callable
from iris.model_wrappers.generative_models.base import GenerativeLLM


class HuggfaceGenerativeLLM(GenerativeLLM):
    def __init__(
            self, 
            huggingface_model_name_or_path: str, 
            system_prompt: str = None, 
            max_tokens: int = None,
            post_processing: Callable = None,
            **kwargs
    ):
        super().__init__(post_processing=post_processing)
        self.llm = transformers.pipeline(
            "text-generation", 
            model=huggingface_model_name_or_path,
            device_map="auto",
            **kwargs
        )
        self.model_name = huggingface_model_name_or_path
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens

    def _complete(self, promt: str, **kwargs) -> str:
        if self.system_prompt:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": promt},
            ]
        else:
            messages = [{"role": "user", "content": promt}]
        answer = self.llm(messages, max_new_tokens=self.max_tokens, **kwargs)[0]["generated_text"][-1]["content"]
        return answer