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
    example_template: str = "Input: {query}\nOutput: {answer}"
    prompt_template: Dict[str, str] = field(default_factory=lambda: {
        "instruction_only": "Instruction: {instruction}\nOutput: ",
        "instruction_with_query": "Instruction: {instruction}\n\nInput: {query}\nOutput: ",
        "instruction_with_examples_and_query": "Instruction: {instruction}\n\n{examples}\n\nInput: {query}\nOutput: ",
        "query_only": "Input: {query}\nOutput: ",
        "query_with_examples": "{examples}\n\nInput: {query}\nOutput: ",
    })

    def get_example_string(self) -> str:
        example_prompts = []
        for example in self.examples:
            example_prompt = self.example_template.format(
                query=example[0],
                answer=example[1],
            )
            example_prompts.append(example_prompt)
        return "\n\n".join(example_prompts)

    def get_prompts(self) -> List[str]:
        prompts = []
        if self.instructions:
            for instruction in self.instructions:
                if len(self.examples) > 0 and self.query:
                    examples = self.get_example_string()
                    prompt = self.prompt_template["instruction_with_examples_and_query"].format(
                        instruction=instruction,
                        examples=examples,
                        query=self.query,
                    )
                elif self.query:
                    prompt = self.prompt_template["instruction_with_query"].format(
                        instruction=instruction,
                        query=self.query,
                    )
                else:
                    prompt = self.prompt_template["instruction_only"].format(
                        instruction=instruction,
                    )
                prompts.append(prompt)
        elif self.query:
            if len(self.examples) > 0:
                examples = self.get_example_string()
                prompt = self.prompt_template["query_with_examples"].format(
                    examples=examples,
                    query=self.query,
                )
            elif self.query:
                prompt = self.prompt_template["query_only"].format(
                    query=self.query,
                )
            prompts.append(prompt)
        else:
            raise ValueError("No query or instructions provided")
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
            example_template=sample.example_template,
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
            example_template=response.example_template,
            prompt_template=response.prompt_template,
            contexts=deepcopy(response.contexts),
            answers=deepcopy(response.answers),
            answer_model=response.answer_model,
        )