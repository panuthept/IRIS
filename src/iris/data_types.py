import numpy as np
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Any
from iris.prompt_template import PromptTemplate


@dataclass
class Sample:
    query: str = None
    instructions: List[str] = field(default_factory=list)
    reference_instruction: str = None   # reference instruction is used for consistency checking
    reference_contexts: List[str] = field(default_factory=list)
    reference_answers: List[str] = field(default_factory=list)
    reference_answers_model: str = None
    examples: List[Tuple[str, str]] = field(default_factory=list)
    example_template: str = "Input: {query}\nOutput: {answer}"
    prompt_template: PromptTemplate = field(default_factory=PromptTemplate)

    def as_dict(self) -> Dict[str, Any]:
        data = {}
        for key, value in self.__dict__.items():
            if key == "prompt_template":
                data[key] = self.prompt_template.as_partial_dict(
                    query=self.query, 
                    instruction=self.instructions[0] if len(self.instructions) > 0 else None, 
                    examples=self.examples if len(self.examples) > 0 else None,
                )
            else:
                if isinstance(value, list) or isinstance(value, dict):
                    if len(value) == 0:
                        continue
                else:
                    if value is None:
                        continue
                data[key] = value
        return data
    
    def get_ref_prompt(self) -> str:
        return self.prompt_template.get_prompt(
            query=self.query, 
            instruction=self.reference_instruction, 
            examples=self.examples if len(self.examples) > 0 else None,
        )

    def get_prompts(self) -> List[str]:
        prompts = []
        if self.instructions:
            for instruction in self.instructions:
                prompt = self.prompt_template.get_prompt(
                    query=self.query, 
                    instruction=instruction, 
                    examples=self.examples if len(self.examples) > 0 else None,
                )
                prompts.append(prompt)
        elif self.query:
            prompt = self.prompt_template.get_prompt(
                query=self.query, 
                examples=self.examples if len(self.examples) > 0 else None,
            )
            prompts.append(prompt)
        return prompts


@dataclass
class ModelResponse(Sample):
    contexts: List[str] = field(default_factory=list)
    answers: List[str] = field(default_factory=list)    # Number of answers is equal to the number of instructions
    answer_model: str = None

    def as_dict(self) -> Dict[str, Any]:
        data = super().as_dict()
        for key, value in self.__dict__.items():
            if key == "prompt_template":
                data[key].update(self.prompt_template.as_partial_dict(
                    query=self.query, 
                    instruction=self.instructions[0] if len(self.instructions) > 0 else None, 
                    examples=self.examples if len(self.examples) > 0 else None,
                    answer=self.answers[0] if len(self.answers) > 0 else None,
                ))
            else:
                if isinstance(value, list) or isinstance(value, dict):
                    if len(value) == 0:
                        continue
                else:
                    if value is None:
                        continue
                data[key] = value
        return data

    def get_prompts(self) -> List[str]:
        prompts = []
        if self.instructions:
            for instruction, answer in zip(self.instructions, self.answers):
                prompt = self.prompt_template.get_prompt(
                    query=self.query, 
                    instruction=instruction, 
                    examples=self.examples if len(self.examples) > 0 else None,
                    answer=answer,
                )
                prompts.append(prompt)
        elif self.query:
            prompt = self.prompt_template.get_prompt(
                query=self.query, 
                examples=self.examples if len(self.examples) > 0 else None,
                answer=self.answers[0],
            )
            prompts.append(prompt)
        return prompts

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
    

@dataclass
class SummarizedResult:
    scores: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_results(cls, results: List[EvaluationResult]):
        summarized_result = cls()

        all_scores = {}
        instruction_scores = {}
        for result in results:
            for metric, value in result.scores.items():
                assert "all" in value.keys(), f"Missing 'all' key in {metric} scores"
                if metric not in all_scores:
                    all_scores[metric] = []
                all_scores[metric].extend(value["all"])

                if metric not in instruction_scores:
                    instruction_scores[metric] = {}
                for idx, score in enumerate(value["all"]):
                    if idx not in instruction_scores[metric]:
                        instruction_scores[metric][idx] = []
                    instruction_scores[metric][idx].append(score)
        instruction_scores = {metric: [np.mean(instruction_scores[metric][idx]) for idx in instruction_scores[metric]] for metric in instruction_scores}

        for metric in instruction_scores:
            summarized_result.scores[metric] = {
                "mean_inst": np.mean(instruction_scores[metric]),
                "std_inst": np.std(instruction_scores[metric]),
                "supports_inst": len(instruction_scores[metric]),
                "mean_all": np.mean(all_scores[metric]),
                "std_all": np.std(all_scores[metric]),
                "supports_all": len(all_scores[metric]),
            }
        return summarized_result