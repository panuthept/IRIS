from typing import List
from copy import deepcopy
from iris.data_types import Sample
from abc import ABC, abstractmethod


class InstructionAugmentation(ABC):
    @abstractmethod
    def augment(self, instruction: str) -> List[str]:
        pass

    def augment_sample(self, sample: Sample) -> Sample:
        assert sample.instructions, "Sample instructions must be provided."
        assert len(sample.instructions) == 1, "Only one instruction is supported."
        sample = deepcopy(sample)

        original_instruction = sample.instructions[0]
        sample.instructions = self.augment(original_instruction)
        sample.reference_instruction = original_instruction
        return sample

    def augment_batch(self, samples: List[Sample]) -> List[Sample]:
        return [self.augment_sample(sample) for sample in samples]