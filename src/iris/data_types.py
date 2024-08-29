from copy import deepcopy
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Any


@dataclass
class Sample:
    query: str = None
    instructions: List[str] = field(default_factory=list)
    reference_contexts: List[str] = field(default_factory=list)
    reference_answers: List[str] = field(default_factory=list)
    reference_answers_model: str = None
    examples: List[Tuple[str, str]] = field(default_factory=list)
    query_template: str = "Input: {query}\nOutput: {answer}"
    prompt_template: str = "Instruction: {instruction}\n\n{examples}\n\n{query}"

    def get_example_string(self) -> str:
        example_prompts = []
        for example in self.examples:
            example_prompt = self.query_template.format(
                query=example[0],
                answer=example[1],
            )
            example_prompts.append(example_prompt)
        return "\n\n".join(example_prompts)

    def get_prompts(self) -> List[str]:
        prompts = []
        for instruction in self.instructions:
            examples = self.get_example_string()
            prompt = self.prompt_template.format(
                instruction=instruction,
                examples=examples,
                query=self.query_template.format(query=self.query, answer=""),
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
            examples=deepcopy(sample.examples),
            query_template=sample.query_template,
            prompt_template=sample.prompt_template,
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
            examples=deepcopy(response.examples),
            query_template=response.query_template,
            prompt_template=response.prompt_template,
            contexts=deepcopy(response.contexts),
            answers=deepcopy(response.answers),
            answer_model=response.answer_model,
        )