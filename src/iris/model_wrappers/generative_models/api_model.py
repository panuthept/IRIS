from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from iris.model_wrappers.generative_models.base import GenerativeLLM


class APIGenerativeLLM(GenerativeLLM):
    def __init__(self, llm: OpenAI, system_prompt: str = None):
        self.llm = llm
        self.model_name = llm.model
        self.system_prompt = system_prompt

    def _complete(self, instruction: str) -> str:
        if self.system_prompt:
            messages = [
                ChatMessage(role="system", content=self.system_prompt),
                ChatMessage(role="user", content=instruction),
            ]
        else:
            messages = [ChatMessage(role="user", content=instruction)]
        answer = self.llm.chat(messages).message.content
        return answer