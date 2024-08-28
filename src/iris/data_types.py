from copy import deepcopy
from typing import List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class Sample:
    instruction: str = None
    instruction_variations: List[str] = field(default_factory=list)
    reference_contexts: List[str] = field(default_factory=list)
    reference_answers: List[str] = field(default_factory=list)
    reference_answers_model: str = None


@dataclass
class ModelResponse(Sample):
    predicted_contexts: List[str] = field(default_factory=list)
    predicted_answer: str = None
    predicted_answer_variations: List[str] = field(default_factory=list)
    predicted_answer_model: str = None

    @classmethod
    def from_sample(
        cls, 
        sample: Sample, 
    ):
        return cls(
            instruction=sample.instruction,
            instruction_variations=deepcopy(sample.instruction_variations),
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
            instruction=response.instruction,
            instruction_variations=deepcopy(response.instruction_variations),
            reference_contexts=deepcopy(response.reference_contexts),
            reference_answers=deepcopy(response.reference_answers),
            reference_answers_model=response.reference_answers_model,
            predicted_contexts=deepcopy(response.predicted_contexts),
            predicted_answer=response.predicted_answer,
            predicted_answer_variations=deepcopy(response.predicted_answer_variations),
            predicted_answer_model=response.predicted_answer_model,
        )