from typing import Callable
from abc import ABC, abstractmethod
# from easyjailbreak.models import ModelBase
from iris.cache import CacheStorage, CacheMode


class LLM:
    def __init__(
            self, 
            system_prompt: str = None, 
            post_processing: Callable = None, 
            max_new_tokens: int = 1024,
            max_tokens: int = 2048,
            temperature: float = 1,
            logprobs: bool = True,
            top_logprobs: int = 10,
            use_cache: bool = False,
            cache_path: str = None,
            cache_mode: CacheMode = CacheMode.ALLOW_DUPLICATE,
            **kwargs,
    ):
        self.system_prompt = system_prompt
        self.post_processing = post_processing
        self.max_new_tokens = max_new_tokens
        self.max_tokens = max_tokens
        self.temperature = temperature
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