from tqdm import tqdm
from typing import List, Callable
from abc import ABC, abstractmethod
from iris.cache import CacheStorage
from iris.data_types import Sample, ModelResponse


class GenerativeLLM(ABC):
    def __init__(
            self, 
            system_prompt: str = None, 
            post_processing: Callable = None, 
            use_cache: bool = True,
            cache_path: str = None,
    ):
        self.system_prompt = system_prompt
        self.post_processing = post_processing
        self.use_cache = use_cache
        self.cache_storage = CacheStorage(self.get_model_name(), cache_path)

    @abstractmethod
    def get_model_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def _complete(self, prompt: str, ref_prompt: str = None, apply_chat_template: bool = True, **kwargs) -> str:
        raise NotImplementedError
    
    def complete(
            self, 
            prompt: str, 
            ref_prompt: str = None,   # reference prompt for TransformerLens
            apply_chat_template: bool = True,
            **kwargs
    ) -> str:
        # Get the answer from cache if available
        answer = None
        if self.use_cache:
            answer = self.cache_storage.retrieve(prompt, system_prompt=self.system_prompt)
        if answer is None:
            answer = self._complete(
                prompt, 
                ref_prompt=ref_prompt, 
                apply_chat_template=apply_chat_template, 
                **kwargs
            )
            # Cache the answer
            self.cache_storage.cache(answer, prompt, system_prompt=self.system_prompt)
        # Post process the answer
        if self.post_processing:
            answer = self.post_processing(answer)
        return answer
    
    def complete_sample(
            self, 
            sample: Sample, 
            apply_chat_template: bool = True,
            **kwargs
    ) -> ModelResponse:
        # Intiial GenerativeLLMResponse
        response = ModelResponse.from_sample(sample)
        # Get the answers
        for prompt in sample.get_prompts():
            ref_prompt = sample.get_ref_prompt()
            answer = self.complete(
                prompt, 
                ref_prompt=ref_prompt, 
                apply_chat_template=apply_chat_template, 
                **kwargs
            )
            response.answers.append(answer)
        # Set the answer model name
        response.answer_model = self.get_model_name()
        return response
    
    def complete_batch(
            self, 
            samples: List[Sample], 
            apply_chat_template: bool = True,
            verbose: bool = True,
            **kwargs
    ) -> List[ModelResponse]:
        return [self.complete_sample(
            sample, 
            apply_chat_template=apply_chat_template, 
            **kwargs
        ) for sample in tqdm(samples, disable=not verbose)]