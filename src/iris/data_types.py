from copy import deepcopy
from typing import List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class Sample:
    query: str = None
    instructions: List[str] = field(default_factory=list)
    reference_contexts: List[str] = field(default_factory=list)
    reference_answers: List[str] = field(default_factory=list)
    reference_answers_model: str = None
    prompt_template: str = "Instruction: {instruction}\n\nQuery: {query}"

    def get_prompts(self, prompt_template: str = None) -> List[str]:
        prompts = []
        for instruction in self.instructions:
            prompt = (prompt_template or self.prompt_template).format(
                instruction=instruction,
                query=self.query,
            )
            prompts.append(prompt)
        return prompts


@dataclass
class ModelResponse(Sample):
    contexts: List[str] = field(default_factory=list)
    answers: List[str] = field(default_factory=list)    # Number of answers is equal to the number of instructions
    answer_model: str = None

    @classmethod
    def from_sample(
        cls, 
        sample: Sample, 
    ):
        return cls(
            query=sample.query,
            instructions=deepcopy(sample.instructions),
            reference_contexts=deepcopy(sample.reference_contexts),
            reference_answers=deepcopy(sample.reference_answers),
            reference_answers_model=sample.reference_answers_model,
        )
    

@dataclass
class EvaluationResult(ModelResponse):
    scores: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_response(cls, response: ModelResponse):
        return cls(
            query=response.query,
            instructions=deepcopy(response.instructions),
            reference_contexts=deepcopy(response.reference_contexts),
            reference_answers=deepcopy(response.reference_answers),
            reference_answers_model=response.reference_answers_model,
            contexts=deepcopy(response.contexts),
            answers=deepcopy(response.answers),
            answer_model=response.answer_model,
        )