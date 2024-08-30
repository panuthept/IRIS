from typing import Callable
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from iris.model_wrappers.generative_models.base import GenerativeLLM


class APIGenerativeLLM(GenerativeLLM):
    def __init__(
            self, 
            llm: OpenAI, 
            system_prompt: str = None, 
            post_processing: Callable = None
    ):
        super().__init__(post_processing=post_processing)
        self.llm = llm
        self.model_name = llm.model
        self.system_prompt = system_prompt

    def _complete(self, promt: str, **kwargs) -> str:
        if self.system_prompt:
            messages = [
                ChatMessage(role="system", content=self.system_prompt),
                ChatMessage(role="user", content=promt),
            ]
        else:
            messages = [ChatMessage(role="user", content=promt)]
        answer = self.llm.chat(messages, **kwargs).message.content
        return answer