import numpy as np
from torch import Tensor
from collections import ChainMap
from dataclasses import dataclass, field
from iris.prompt_template import PromptTemplate
from typing import Union, Tuple, List, Dict, Any


def all_annotations(cls) -> ChainMap:
    """
    Returns a dictionary-like ChainMap that includes annotations for all 
    attributes defined in cls or inherited from superclasses.
    Credit: https://stackoverflow.com/questions/63903901/how-can-i-access-to-annotations-of-parent-class
    """
    return ChainMap(*(c.__annotations__ for c in cls.__mro__ if '__annotations__' in c.__dict__) )


@dataclass
class IRISConfig:
    mode: str = "fixed"
    alpha: float = 0.1
    wait_steps: int = 0
    ema_alpha: float = 0.1
    sma_window_size: int = 10
    label_smoothing: float = 0.0
    layer_labels: Dict[str, Dict[int, int]] = field(default_factory=dict)
    layer_weights: Dict[str, Dict[int, float]] = field(default_factory=dict)
    freeze_layers: List[str] = field(default_factory=list)


@dataclass
class IRISL2Config:
    alpha: float = 0.1
    layer_labels: Dict[str, Dict[int, Tensor]] = field(default_factory=dict)
    layer_weights: Dict[str, Dict[int, float]] = field(default_factory=dict)
    freeze_layers: List[str] = field(default_factory=list)


@dataclass
class IRISCLConfig(IRISL2Config):
    pass


@dataclass
class Sample:
    query: str = None
    instructions: List[str] = field(default_factory=list)
    instructions_pred_label: List[str] = field(default_factory=list)   # 'harmful' or 'benign'
    instructions_true_label: List[str] = field(default_factory=list)   # 'harmful' or 'benign'
    reference_instruction: str = None                             # Reference instruction is used for consistency checking
    reference_contexts: List[str] = field(default_factory=list)
    reference_answers: List[str] = field(default_factory=list)    # Reference answers can be used for evaluation both answers and classsified_instructions
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
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Sample':
        # Filter out None values and unknown keys
        data = {k: v for k, v in data.items() if v is not None and k in all_annotations(cls)}
        # If prompt_template is present, convert it to PromptTemplate object
        if "prompt_template" in data:
            data["prompt_template"] = PromptTemplate.from_dict(data["prompt_template"])
        return cls(**data)
    
    def get_ref_prompt(self) -> str:
        return self.prompt_template.get_prompt(
            query=self.query, 
            instruction=self.reference_instruction, 
            examples=self.examples if len(self.examples) > 0 else None,
        ) if self.reference_instruction is not None else None

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
    answers_pred_label: List[str] = field(default_factory=list)
    answers_true_label: List[str] = field(default_factory=list)
    answer_model: str = None

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
    ) -> 'ModelResponse':
        data = sample.as_dict()
        return cls.from_dict(data)
    

@dataclass
class EvaluationResult(ModelResponse):
    scores: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_response(
        cls, 
        response: ModelResponse, 
    ) -> 'EvaluationResult':
        data = response.as_dict()
        return cls.from_dict(data)
    

@dataclass
class SummarizedResult:
    scores: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_results(cls, results: List[EvaluationResult]) -> 'SummarizedResult':
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