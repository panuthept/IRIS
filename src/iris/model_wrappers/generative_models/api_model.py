from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from iris.model_wrappers.generative_models.base import GenerativeLLM


class APIGenerativeLLM(GenerativeLLM):
    def __init__(
            self, 
            llm: OpenAI, 
            **kwargs,
    ):
        self.llm = llm
        super().__init__(**kwargs)

    def get_model_name(self) -> str:
        return self.llm.model

    def _complete(self, prompt: str, ref_prompt: str = None, apply_chat_template: bool = True, **kwargs) -> str:
        if apply_chat_template:
            if self.system_prompt:
                messages = [
                    ChatMessage(role="system", content=self.system_prompt),
                    ChatMessage(role="user", content=prompt),
                ]
            else:
                messages = [ChatMessage(role="user", content=prompt)]
            answer = self.llm.chat(messages, **kwargs).message.content
        else:
            if self.system_prompt:
                prompt = f"{self.system_prompt}\n\n{prompt}"
            answer = self.llm.complete(prompt, **kwargs)
        return answer