import os
import json
from datetime import datetime
from collections import defaultdict
from iris.consts.default_paths import CACHE_STORAGE_DEFAULT_PATH


class CacheStorage:
    temp_file_name = "temp.json"
    cache_file_name = "cache.json"

    def __init__(self, name: str, cache_path: str = None):
        self.name = name
        self.cache_path = cache_path if cache_path else CACHE_STORAGE_DEFAULT_PATH
        self.storage = self._load_storage()

    def _load_storage(self):
        if os.path.exists(os.path.join(self.cache_path, self.name, self.cache_file_name)):
            with open(os.path.join(self.cache_path, self.name, self.cache_file_name), "r") as f:
                return json.load(f)
        else:
            return defaultdict(dict)
        
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

        if system_prompt in self.storage:
            data = self.storage[system_prompt].get(prompt, None)
            if data:
                return data["content"]
        return None

    def cache(self, response: str, prompt: str, system_prompt: str = None):
        if system_prompt is None:
            system_prompt = ""

        if system_prompt not in self.storage:
            self.storage[system_prompt] = {}
            
        self.storage[system_prompt][prompt] = {"content": response, "timestamp": str(datetime.now())}
        self._save_storage()