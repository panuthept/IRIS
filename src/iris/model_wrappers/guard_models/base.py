from typing import List
from copy import deepcopy
from abc import abstractmethod
from iris.model_wrappers import LLM
from iris.data_types import Sample, ModelResponse


class GuardLLM(LLM):
    @abstractmethod
    def _prompt_classify(self, prompt: str) -> str:
        raise NotImplementedError
    
    @abstractmethod
    def _response_classify(self, prompt: str, response: str) -> str:
        raise NotImplementedError

    def prompt_classify(self, sample: Sample) -> Sample:
        sample = deepcopy(sample)
        sample.instructions_pred_label = [self._prompt_classify(inst) for inst in sample.instructions]
        return sample

    def response_classify(self, response: ModelResponse) -> ModelResponse:
        response = deepcopy(response)
        response.answers_pred_label = [self._response_classify(inst, ans) for inst, ans in zip(response.instructions, response.answers)]
        return response
    
    def prompt_classify_batch(self, samples: List[Sample]) -> List[Sample]:
        return [self.prompt_classify(sample) for sample in samples]
    
    def response_classify_batch(self, responses: List[ModelResponse]) -> List[ModelResponse]:
        return [self.response_classify(response) for response in responses]