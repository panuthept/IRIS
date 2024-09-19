import os
import json
import random
from enum import Enum
from typing import List
from datetime import datetime
from collections import defaultdict
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
        return [mode.name for mode in CacheMode]

    def _load_storage(self):
        if os.path.exists(os.path.join(self.cache_path, self.name, self.cache_file_name)):
            with open(os.path.join(self.cache_path, self.name, self.cache_file_name), "r") as f:
                return json.load(f)
        else:
            return defaultdict(dict)

    def _init_session_memory(self):
        return {system_prompt: {prompt: {"seen_content": set()} for prompt in self.storage[system_prompt]} for system_prompt in self.storage}
        
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

    def retrieve(self, prompt: str, system_prompt: str = None) -> str:
        if system_prompt is None:
            system_prompt = ""

        retrieved_content = None
        if system_prompt in self.storage:
            data = self.storage[system_prompt].get(prompt, None)
            if data:
                # Check if data is stored in old format and convert to new format
                if isinstance(data["content"], str):
                    data["content"] = [data["content"]]
                    data["timestamp"] = [data["timestamp"]]

                # Get available contents
                if self.cache_mode == CacheMode.ALLOW_DUPLICATE:
                    available_contents = data["content"]
                elif self.cache_mode == CacheMode.NO_DUPLICATE:
                    available_contents = list(set(data["content"]) - self.session_memory[system_prompt][prompt]["seen_content"])
                else:
                    raise ValueError(f"Invalid cache_mode: {self.cache_mode}")

                # Retrieve content if available
                if len(available_contents) > 0:
                    retrieved_content = random.choice(available_contents)

        if retrieved_content is not None:
            # Update session memory
            self.session_memory[system_prompt][prompt]["seen_content"].add(retrieved_content)
        return retrieved_content

    def cache(self, response: str, prompt: str, system_prompt: str = None):
        if system_prompt is None:
            system_prompt = ""

        if system_prompt not in self.storage:
            self.storage[system_prompt] = {}
            
        if prompt not in self.storage[system_prompt]:
            self.storage[system_prompt][prompt] = {"content": [], "timestamp": []}

        # Check if data is stored in old format and convert to new format
        if isinstance(self.storage[system_prompt][prompt]["content"], str):
            self.storage[system_prompt][prompt]["content"] = [self.storage[system_prompt][prompt]["content"]]
        if isinstance(self.storage[system_prompt][prompt]["timestamp"], str):
            self.storage[system_prompt][prompt]["timestamp"] = [self.storage[system_prompt][prompt]["timestamp"]]

        if response not in self.storage[system_prompt][prompt]["content"]:
            # Add to storage
            self.storage[system_prompt][prompt]["content"].append(response)
            self.storage[system_prompt][prompt]["timestamp"].append(str(datetime.now()))
            # Update session memory
            self.session_memory[system_prompt][prompt]["seen_content"].add(response)
        self._save_storage()