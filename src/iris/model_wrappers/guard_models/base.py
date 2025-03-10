from tqdm import tqdm
from copy import deepcopy
from abc import abstractmethod
from typing import List, Optional
from iris.model_wrappers import LLM
from iris.data_types import Sample, ModelResponse


class GuardLLM(LLM):
    @abstractmethod
    def _prompt_classify(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError
    
    @abstractmethod
    def _response_classify(self, prompt: str, response: str, **kwargs) -> str:
        raise NotImplementedError

    def prompt_classify(self, sample: Sample) -> Sample:
        sample = deepcopy(sample)
        sample.instructions_pred_label = [self._prompt_classify(inst) for inst in sample.instructions]
        return sample

    def response_classify(self, response: ModelResponse) -> ModelResponse:
        response = deepcopy(response)
        response.answers_pred_label = [self._response_classify(inst, ans) for inst, ans in zip(response.instructions, response.answers)]
        return response
    
    def prompt_classify_batch(self, samples: List[Sample], verbose: bool = True) -> List[Sample]:
        return [self.prompt_classify(sample) for sample in tqdm(samples, disable=not verbose)]
    
    def response_classify_batch(self, responses: List[ModelResponse], verbose: bool = True) -> List[ModelResponse]:
        return [self.response_classify(response) for response in tqdm(responses, disable=not verbose)]
    
    def generate(self, *args, **kwargs):
        return self._prompt_classify(*args, **kwargs)
    
    def predict(self, prompt: str, response: Optional[str] = None):
        pass

    @abstractmethod
    def _prompt_classification_tokenizer(self, prompt: str) -> dict:
        raise NotImplementedError
    
    @abstractmethod
    def _response_classification_tokenizer(self, prompt: str, response: str) -> dict:
        raise NotImplementedError

    def tokenize(self, prompt: str, prompt_label: Optional[str] = None, response: Optional[str] = None):
        if prompt_label is not None and response is not None:
            return self._response_classification_tokenizer(prompt, prompt_label, response)
        return self._prompt_classification_tokenizer(prompt)