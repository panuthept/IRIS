from typing import List, Callable
from abc import ABC, abstractmethod
from iris.data_types import Sample, ModelResponse


class GenerativeLLM(ABC):
    model_name = "GenerativeLLM"

    def __init__(self, post_processing: Callable = None):
        self.post_processing = post_processing

    @abstractmethod
    def _complete(self, promt: str) -> str:
        raise NotImplementedError
    
    def complete(self, sample: Sample) -> ModelResponse:
        # Intiial GenerativeLLMResponse
        response = ModelResponse.from_sample(sample)
        # Get the answers
        for promt in sample.get_prompts():
            response.answers.append(self.post_processing(self._complete(promt)))
        # Set the answer model name
        response.answer_model = self.model_name
        return response
    
    def complete_batch(self, samples: List[Sample]) -> List[ModelResponse]:
        return [self.complete(sample) for sample in samples]