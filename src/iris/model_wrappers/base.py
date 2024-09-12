from typing import Callable
from abc import ABC, abstractmethod
from iris.cache import CacheStorage


class LLM(ABC):
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
    def generate(self, *args, **kwargs):
        raise NotImplementedError