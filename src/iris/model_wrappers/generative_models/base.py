import time
from tqdm import tqdm
from abc import abstractmethod
from iris.model_wrappers import LLM
from typing import List, Tuple, Optional
from iris.data_types import Sample, ModelResponse


class GenerativeLLM(LLM):
    @abstractmethod
    def _complete(
        self, 
        prompt: str, 
        ref_prompt: Optional[str] = None, 
        suffix_prompt: Optional[str] = None, 
        apply_chat_template: bool = True, 
        **kwargs
    ) -> Tuple[str, Optional[List[List[Tuple[str, float]]]]]:
        raise NotImplementedError
    
    def complete(
        self, 
        prompt: str, 
        ref_prompt: Optional[str] = None,   # reference prompt for TransformerLens
        suffix_prompt: Optional[str] = None,  # suffix prompt for GuardLLM
        apply_chat_template: bool = True,
        max_trials: int = 1,
        failure_sleep_time: int = 1,
        return_logprobs: bool = False,
        **kwargs
    ) -> str:
        # Get the answer from cache if available
        answer = None
        logprobs = None
        if self.use_cache:
            answer, logprobs = self.cache_storage.retrieve(
                prompt, 
                temperature=self.temperature,
                system_prompt=self.system_prompt,
                apply_chat_template=apply_chat_template,
                max_new_tokens=self.max_new_tokens,
                top_logprobs=self.top_logprobs,
                suffix_prompt=suffix_prompt,
                return_logprobs=self.logprobs,
            )
        if answer is None or (self.logprobs and logprobs is None):
            for _ in range(max_trials):
                try:                
                    answer, logprobs = self._complete(
                        prompt, 
                        ref_prompt=ref_prompt, 
                        suffix_prompt=suffix_prompt,
                        apply_chat_template=apply_chat_template, 
                        **kwargs
                    )
                    break
                except Exception as e:
                    print(f"Failed to generate response: {e}")
                    time.sleep(failure_sleep_time)
            # Cache the answer
            self.cache_storage.cache(
                answer, 
                prompt, 
                temperature=self.temperature,
                system_prompt=self.system_prompt,
                apply_chat_template=apply_chat_template,
                max_new_tokens=self.max_new_tokens,
                logprobs=logprobs,
                suffix_prompt=suffix_prompt,
            )
        # Post process the answer
        if self.post_processing:
            answer = self.post_processing(answer)

        if return_logprobs:
            return answer, logprobs
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