from typing import List, Dict
from abc import ABC, abstractmethod
from iris.prompt_template import PromptTemplate
from iris.data_types import Sample, ModelResponse, SafeGuardInput


class Dataset(ABC):
    def __init__(self, path: str):
        self.data: Dict[str, List] = self._load_dataset(path)

    @classmethod
    def split_available(cls) -> List[str]:
        return ["train", "dev", "test"]

    @abstractmethod
    def _load_dataset(self, path: str) -> Dict[str, List]:
        raise NotImplementedError
    
    @abstractmethod
    def get_size(self, split: str = None) -> int:
        raise NotImplementedError

    @abstractmethod
    def as_samples(self, split: str = None, prompt_template: PromptTemplate = None) -> List[Sample]:
        raise NotImplementedError
    
    def as_dataloader(self, split: str = None, prompt_template: PromptTemplate = None):
        raise NotImplementedError


class JailbreakDataset(Dataset):
    def __init__(
        self, 
        path: str,
        intention: str = None,
        category: str = None,
        language: str = "en",
        attack_engine: str = None,
    ):
        if intention:
            assert intention in self.intentions_available(), f"Intention {intention} not available"
        if category:
            assert category in self.categories_available(), f"Category {category} not available"
        if attack_engine:
            assert attack_engine in self.attack_engines_available(), f"Attack engine {attack_engine} not available"
        assert language in self.language_available(), f"Language {language} not available"
        
        self.language = language
        self.intention = intention
        self.category = category
        self.attack_engine = attack_engine
        self.default_prompt_template = PromptTemplate(
            instruction_template="{instruction}",
            instruction_answer_template="{instruction}\n\n{answer}",
        )
        super().__init__(path)

    @classmethod
    def intentions_available(cls) -> List[str]:
        return ["harmful", "benign"]
    
    @classmethod
    def language_available(cls) -> List[str]:
        return ["en"]
    
    @classmethod
    def categories_available(cls) -> List[str]:
        raise NotImplementedError
    
    @classmethod
    def attack_engines_available(cls) -> List[str]:
        raise NotImplementedError
    
    def get_size(self, split: str = "test") -> int:
        assert split in self.split_available(), f"Split {split} not available"
        return len(self.data[split])
    
    def as_samples(self, split: str = "test") -> List[SafeGuardInput]:
        assert split in self.split_available(), f"Split {split} not available"

        samples: List[SafeGuardInput] = []
        for datum in self.data[split]:
            sample = SafeGuardInput(**{key: value for key, value in datum.items() if key in SafeGuardInput.__annotations__})
            samples.append(sample)
        return samples
    
    # def as_samples(self, split: str = "test", prompt_template: PromptTemplate = None) -> List[Sample]:
    #     assert split in self.split_available(), f"Split {split} not available"

    #     samples: List[Sample] = []
    #     for datum in self.data[split]:
    #         sample = Sample(**{key: value for key, value in datum.items() if key in Sample.__annotations__})
    #         sample.prompt_template = self.default_prompt_template if prompt_template is None else prompt_template
    #         samples.append(sample)
    #     return samples
    
    # def as_responses(self, split="test", prompt_template: PromptTemplate = None) -> List[ModelResponse]:
    #     assert split in self.split_available(), f"Split {split} not available"

    #     responses: List[ModelResponse] = []
    #     for datum in self.data[split]:
    #         response = ModelResponse(**{key: value for key, value in datum.items() if key in ModelResponse.__annotations__ or key in Sample.__annotations__})
    #         response.prompt_template = self.default_prompt_template if prompt_template is None else prompt_template
    #         responses.append(response)
    #     return responses