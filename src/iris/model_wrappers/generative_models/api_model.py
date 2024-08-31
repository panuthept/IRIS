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