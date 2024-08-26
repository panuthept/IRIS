from typing import List
from dataclasses import dataclass


@dataclass
class Sample:
    instruction: str
    responses: List[str]
    references: List[str]
    scores: List[float]

    target_model: str
    reference_model: str