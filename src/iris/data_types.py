from typing import List
from dataclasses import dataclass


@dataclass
class Sample:
    instruction: str
    references: List[str]
    reference_model: str


@dataclass
class Response(Sample):
    responses: List[str]
    target_model: str

    @classmethod
    def from_sample(cls, sample: Sample, responses: List[str], target_model: str):
        return cls(
            instruction=sample.instruction,
            references=sample.references,
            reference_model=sample.reference_model,
            responses=responses,
            target_model=target_model,
        )


@dataclass
class Result(Response):
    scores: List[float]