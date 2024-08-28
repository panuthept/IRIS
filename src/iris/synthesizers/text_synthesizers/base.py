import json
from typing import List
from llama_index.core import PromptTemplate
from llama_index.core.query_pipeline import QueryPipeline


class BaseTextSynthesizer:
    """Base class for text synthesizers."""
    prompt_template = (
        "Given an input text in ``` ``` format, {instruction}\n"
        "Generate structured JSON format with a 'output' key and list value.\n"
        "```{text}```\n"
    )

    def __init__(self, llm):
        self.llm = llm
        self.instruction = ""
        self.system_prompt = PromptTemplate(self.prompt_template)
        self.pipeline = QueryPipeline(chain=[self.system_prompt, self.llm])

    def synthesize(self, text: str) -> List[str]:
        response = self.pipeline.run(text=text, instruction=self.instruction).message.content
        response = response.replace("```json", "").replace("```", "")
        response = json.loads(response)
        return response["output"]

    def synthesize_batch(self, texts: List[str]) -> List[List[str]]:
        return [self.synthesize(text) for text in texts]