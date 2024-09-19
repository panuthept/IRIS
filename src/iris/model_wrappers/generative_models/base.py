import time
from tqdm import tqdm
from typing import List
from abc import abstractmethod
from iris.model_wrappers import LLM
from iris.data_types import Sample, ModelResponse


class GenerativeLLM(LLM):
    @abstractmethod
    def _complete(self, prompt: str, ref_prompt: str = None, apply_chat_template: bool = True, **kwargs) -> str:
        raise NotImplementedError
    
    def complete(
            self, 
            prompt: str, 
            ref_prompt: str = None,   # reference prompt for TransformerLens
            apply_chat_template: bool = True,
            max_trials: int = 1,
            failure_sleep_time: int = 1,
            **kwargs
    ) -> str:
        # Get the answer from cache if available
        answer = None
        if self.use_cache:
            answer = self.cache_storage.retrieve(prompt, system_prompt=self.system_prompt)
        if answer is None:
            for _ in range(max_trials):
                try:                
                    answer, logprobs = self._complete(
                        prompt, 
                        ref_prompt=ref_prompt, 
                        apply_chat_template=apply_chat_template, 
                        **kwargs
                    )
                    print(logprobs)
                    break
                except Exception as e:
                    print(f"Failed to generate response: {e}")
                    time.sleep(failure_sleep_time)
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
    
    def generate(self, *args, **kwargs):
        return self.complete(*args, **kwargs)