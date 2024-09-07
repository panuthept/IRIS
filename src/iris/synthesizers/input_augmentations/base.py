import json
from typing import List
from iris.data_types import Sample
from llama_index.core import PromptTemplate
from llama_index.core.query_pipeline import QueryPipeline


class TextSynthesizer:
    """Base class for text synthesizers."""
    prompt_template = (
        "Given an input text in ``` ``` format, {instruction}\n"
        "Generate structured JSON format with a 'output' key and list value.\n"
        "```{text}```\n"
    )

    def __init__(self, llm, instruction: str = ""):
        self.llm = llm
        self.instruction = instruction
        self.system_prompt = PromptTemplate(self.prompt_template)
        self.pipeline = QueryPipeline(chain=[self.system_prompt, self.llm])

    def _synthesize(self, text: str) -> List[str]:
        response = self.pipeline.run(text=text, instruction=self.instruction).message.content
        response = response.replace("```json", "").replace("```", "")
        response = json.loads(response)
        return response["output"]

    def synthesize(self, sample: Sample) -> Sample:
        sample.instruction_variations = self._synthesize(sample.instruction)
        return sample

    def synthesize_batch(self, samples: List[Sample]) -> List[List[Sample]]:
        return [self.synthesize(sample) for sample in samples]