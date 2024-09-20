import os
import json
import random
from enum import Enum
from datetime import datetime
from collections import defaultdict
from typing import List, Tuple, Optional
from iris.consts.default_paths import CACHE_STORAGE_DEFAULT_PATH


class CacheMode(Enum):
    ALLOW_DUPLICATE = 0
    NO_DUPLICATE = 1


class CacheStorage:
    temp_file_name = "temp.json"
    cache_file_name = "cache.json"

    def __init__(
            self, 
            name: str, 
            cache_path: str = None, 
            cache_mode: CacheMode = CacheMode.ALLOW_DUPLICATE,
    ):
        """
        self.storage = {
            String: {
                String: {
                    "content": String|List[String],     # String is deprecated but retained for backward compatibility
                    "timestamp": String|List[String],   # String is deprecated but retained for backward compatibility
                    "logprobs": List[List[Tuple[String, float]]],
                }
            }
        }
        self.session_memory = {
            String: {
                String: {
                    "seen_content": Set[String],
                }
            }
        }
        """
        assert cache_mode in self.mode_available(), f"cache_mode must be one of {self.mode_available()}"
        self.name = name
        self.cache_path = cache_path if cache_path else CACHE_STORAGE_DEFAULT_PATH
        self.cache_mode = cache_mode
        self.storage = self._load_storage()
        self.session_memory = self._init_session_memory()

    @classmethod
    def mode_available(cls) -> List[str]:
        return [mode for mode in CacheMode]

    def _load_storage(self):
        if os.path.exists(os.path.join(self.cache_path, self.name, self.cache_file_name)):
            with open(os.path.join(self.cache_path, self.name, self.cache_file_name), "r") as f:
                return json.load(f)
        else:
            return defaultdict(dict)

    def _init_session_memory(self):
        return {system_prompt: {prompt: {"seen_content": set()} for prompt in self.storage[system_prompt]} for system_prompt in self.storage}
    
    def _update_session_memory(self, system_prompt: str, prompt: str, response: str):
        if system_prompt not in self.session_memory:
            self.session_memory[system_prompt] = {}
        if prompt not in self.session_memory[system_prompt]:
            self.session_memory[system_prompt][prompt] = {"seen_content": set()}
        self.session_memory[system_prompt][prompt]["seen_content"].add(response)
        
    def _save_storage(self):
        # Create the cache directory if it does not exist
        if not os.path.exists(os.path.join(self.cache_path, self.name)):
            os.makedirs(os.path.join(self.cache_path, self.name))

        # Save to temp file first
        with open(os.path.join(self.cache_path, self.name, self.temp_file_name), "w") as f:
            json.dump(self.storage, f, indent=4)
        # Move to the actual file
        os.rename(
            os.path.join(self.cache_path, self.name, self.temp_file_name),
            os.path.join(self.cache_path, self.name, self.cache_file_name),
        )
        # Remove the temp file
        if os.path.exists(os.path.join(self.cache_path, self.name, self.temp_file_name)):
            os.remove(os.path.join(self.cache_path, self.name, self.temp_file_name))

    @staticmethod
    def _get_key(
        prompt: str, 
        apply_chat_template: Optional[bool] = None,
        max_new_tokens: Optional[int] = None,
        suffix_prompt: Optional[str] = None,
    ) -> str:
        if apply_chat_template is None and max_new_tokens is None:
            # NOTE: This is for backward compatibility
            return prompt
        
        key = f"prompt: {prompt}"
        if apply_chat_template is not None:
            key = f"apply_chat_template: {apply_chat_template}, {key}"
        if max_new_tokens is not None:
            key = f"max_new_tokens: {max_new_tokens}, {key}"
        if suffix_prompt is not None:
            key = f"suffix_prompt: {suffix_prompt}, {key}"
        return key

    def retrieve(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None, 
        apply_chat_template: Optional[bool] = None,
        max_new_tokens: Optional[int] = None,
        top_logprobs: Optional[int] = None,
        suffix_prompt: Optional[str] = None,
        return_logprobs: bool = False,
    ) -> Tuple[str, Optional[List[List[Tuple[str, float]]]]]:
        if system_prompt is None:
            system_prompt = ""
        key = self._get_key(
            prompt, 
            apply_chat_template=apply_chat_template,
            max_new_tokens=max_new_tokens,
            suffix_prompt=suffix_prompt,
        )

        logprobs = None
        retrieved_content = None
        if system_prompt in self.storage:
            data = self.storage[system_prompt].get(key, None)
            if data:
                # NOTE: This is for backward compatibility
                # Check if data is stored in old format and convert to new format
                if isinstance(data["content"], str):
                    data["content"] = [data["content"]]
                    data["timestamp"] = [data["timestamp"]]

                # Get available contents
                if self.cache_mode == CacheMode.ALLOW_DUPLICATE:
                    available_contents = data["content"]
                elif self.cache_mode == CacheMode.NO_DUPLICATE:
                    available_contents = list(set(data["content"]) - self.session_memory[system_prompt][key]["seen_content"])
                else:
                    raise ValueError(f"Invalid cache_mode: {self.cache_mode}")

                # Retrieve content if available
                if len(available_contents) > 0:
                    retrieved_content = random.choice(available_contents)

                # Get logprobs
                if return_logprobs:
                    logprobs = data.get("logprobs", None)
                    # Verify logprobs
                    if logprobs is not None:
                        if top_logprobs is not None:
                            logprobs = [logprob[:top_logprobs] for logprob in logprobs]
                            if not all([len(logprob) >= top_logprobs for logprob in logprobs]):
                                logprobs = None

        if retrieved_content is not None:
            # Update session memory
            self._update_session_memory(system_prompt, key, retrieved_content)
        return retrieved_content, logprobs

    def cache(
        self, 
        response: str, 
        prompt: str, 
        system_prompt: Optional[str] = None, 
        apply_chat_template: Optional[bool] = None,
        max_new_tokens: Optional[int] = None,
        logprobs: Optional[List[List[Tuple[str, float]]]] = None,
        suffix_prompt: Optional[str] = None,
    ):
        if system_prompt is None:
            system_prompt = ""
        key = self._get_key(
            prompt, 
            apply_chat_template=apply_chat_template,
            max_new_tokens=max_new_tokens,
            suffix_prompt=suffix_prompt,
        )

        if system_prompt not in self.storage:
            self.storage[system_prompt] = {}
            
        if key not in self.storage[system_prompt]:
            self.storage[system_prompt][key] = {"content": [], "timestamp": [], "logprobs": None}

        # NOTE: This is for backward compatibility
        # Check if data is stored in old format and convert to new format
        if isinstance(self.storage[system_prompt][key]["content"], str):
            self.storage[system_prompt][key]["content"] = [self.storage[system_prompt][key]["content"]]
        if isinstance(self.storage[system_prompt][key]["timestamp"], str):
            self.storage[system_prompt][key]["timestamp"] = [self.storage[system_prompt][key]["timestamp"]]
        if "logprobs" not in self.storage[system_prompt][key]:
            self.storage[system_prompt][key]["logprobs"] = None

        if response not in self.storage[system_prompt][key]["content"]:
            # Add to storage
            self.storage[system_prompt][key]["content"].append(response)
            self.storage[system_prompt][key]["timestamp"].append(str(datetime.now()))
            # Update session memory
            self._update_session_memory(system_prompt, key, response)
            
        if logprobs is not None:
            self.storage[system_prompt][key]["logprobs"] = logprobs
        self._save_storage()