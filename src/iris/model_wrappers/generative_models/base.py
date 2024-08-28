from typing import List
from abc import ABC, abstractmethod
from iris.data_types import Sample, GenerativeLLMResponse


class GenerativeLLM(ABC):
    @abstractmethod
    def _complete(self, instruction: str) -> str:
        raise NotImplementedError
    
    def complete(self, sample: Sample) -> GenerativeLLMResponse:
        assert sample.instruction, "Instruction is required for completion."

        # Intiial GenerativeLLMResponse
        response = GenerativeLLMResponse.from_sample(sample)
        # Get the answer
        response.answer = self._complete(sample.instruction)
        # Get the answer variations if any (optional)
        if sample.instruction_variations:
            response.answer_variations = [self._complete(instruction_variation) for instruction_variation in sample.instruction_variations]
        # Set the answer model name
        response.answer_model = self.model_name
        return response
    
    def complete_batch(self, samples: List[Sample]) -> List[GenerativeLLMResponse]:
        return [self.complete(sample) for sample in samples]