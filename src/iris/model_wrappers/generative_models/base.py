from typing import List
from abc import ABC, abstractmethod
from iris.data_types import Sample, ModelResponse


class GenerativeLLM(ABC):
    model_name = "GenerativeLLM"

    @abstractmethod
    def _complete(self, instruction: str) -> str:
        raise NotImplementedError
    
    def complete(self, sample: Sample) -> ModelResponse:
        assert sample.instruction, "Instruction is required for completion."

        # Intiial GenerativeLLMResponse
        response = ModelResponse.from_sample(sample)
        # Get the answer
        response.predicted_answer = self._complete(sample.instruction)
        # Get the answer variations if any (optional)
        if sample.instruction_variations:
            response.predicted_answer_variations = [self._complete(instruction_variation) for instruction_variation in sample.instruction_variations]
        # Set the answer model name
        response.predicted_answer_model = self.model_name
        return response
    
    def complete_batch(self, samples: List[Sample]) -> List[ModelResponse]:
        return [self.complete(sample) for sample in samples]