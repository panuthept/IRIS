from typing import List
from copy import deepcopy
from dataclasses import dataclass


@dataclass
class Sample:
    instruction: str = None
    instruction_variations: List[str] = None

    references: List[str] = None
    reference_model: str = None


@dataclass
class GenerativeLLMResponse(Sample):
    answer: str = None
    answer_variations: List[str] = None
    answer_model: str = None

    @classmethod
    def from_sample(
        cls, 
        sample: Sample, 
    ):
        return cls(
            instruction=sample.instruction,
            instruction_variations=deepcopy(sample.instruction_variations),
            references=deepcopy(sample.references),
            reference_model=sample.reference_model,
        )


@dataclass
class GenerativeLLMResult(GenerativeLLMResponse):
    answer_score: float = None
    consistency_score: float = None

    @classmethod
    def from_response(cls, response: GenerativeLLMResponse):
        return cls(
            instruction=response.instruction,
            instruction_variations=deepcopy(response.instruction_variations),
            references=deepcopy(response.references),
            reference_model=response.reference_model,
            answer=deepcopy(response.answer),
            answer_variations=deepcopy(response.answer_variations),
            answer_model=response.answer_model,
        )