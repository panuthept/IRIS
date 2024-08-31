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