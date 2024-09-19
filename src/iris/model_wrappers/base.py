from typing import Callable
from abc import ABC, abstractmethod
from easyjailbreak.models import ModelBase
from iris.cache import CacheStorage, CacheMode


class LLM(ModelBase):
    def __init__(
            self, 
            system_prompt: str = None, 
            post_processing: Callable = None, 
            use_cache: bool = True,
            cache_path: str = None,
            cache_mode: CacheMode = CacheMode.ALLOW_DUPLICATE,
            **kwargs,
    ):
        self.system_prompt = system_prompt
        self.post_processing = post_processing
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