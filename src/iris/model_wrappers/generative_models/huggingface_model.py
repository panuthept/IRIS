import transformers
from iris.model_wrappers.generative_models.base import GenerativeLLM


class HuggfaceGenerativeLLM(GenerativeLLM):
    def __init__(
            self, 
            huggingface_model_name_or_path: str,  
            max_tokens: int = None,
            pipeline_kwargs: dict = None,
            **kwargs,
    ):
        # TODO: transformers.pipeline does not allow batch generation. We need to find a way to generate multiple responses at once
        self.llm = transformers.pipeline(
            "text-generation", 
            model=huggingface_model_name_or_path,
            device_map="auto",
            **pipeline_kwargs
        )
        self.model_name = huggingface_model_name_or_path
        self.max_tokens = max_tokens
        super().__init__(**kwargs)

    def get_model_name(self) -> str:
        # TODO: Add a better way to get the model name. the current way is not reliable as the huggingface_model_name_or_path can be a path
        return self.model_name

    def _complete(self, prompt: str, ref_prompt: str = None, apply_chat_template: bool = True, **kwargs) -> str:
        if apply_chat_template:
            if self.system_prompt:
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ]
            else:
                messages = [{"role": "user", "content": prompt}]
            answer = self.llm(messages, max_new_tokens=self.max_tokens, **kwargs)[0]["generated_text"][-1]["content"]
        else:
            if self.system_prompt:
                prompt = f"{self.system_prompt}\n\n{prompt}"
            completed_text = self.llm(prompt, max_length=self.max_tokens, **kwargs)[0]["generated_text"]
            answer = completed_text[len(prompt):]
        return answer