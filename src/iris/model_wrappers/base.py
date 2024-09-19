from typing import Callable
from abc import ABC, abstractmethod
from easyjailbreak.models import ModelBase
from iris.cache import CacheStorage, CacheMode


class LLM(ModelBase):
    def __init__(
            self, 
            system_prompt: str = None, 
            post_processing: Callable = None, 
            max_new_tokens: int = 1024,
            logprobs: bool = True,
            top_logprobs: int = 10,
            use_cache: bool = True,
            cache_path: str = None,
            cache_mode: CacheMode = CacheMode.ALLOW_DUPLICATE,
            **kwargs,
    ):
        self.system_prompt = system_prompt
        self.post_processing = post_processing
        self.max_new_tokens = max_new_tokens
        self.logprobs = logprobs
        self.top_logprobs = top_logprobs
        self.use_cache = use_cache
        self.cache_storage = CacheStorage(
            name=self.get_model_name(), 
            cache_path=cache_path, 
            cache_mode=cache_mode
        )

    @abstractmethod
    def get_model_name(self) -> str:
        raise NotImplementedError
    
    @abstractmethod
    def generate(self, *args, **kwargs):
        raise NotImplementedError