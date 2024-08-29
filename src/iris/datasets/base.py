from typing import List
from iris.data_types import Sample
from abc import ABC, abstractmethod


class Dataset(ABC):
    def __init__(self, path: str):
        self.path = path
        self.inputs = None
        self.labels = None

    @abstractmethod
    def _load_dataset(self, path: str) -> List[Sample]:
        raise NotImplementedError

    def as_samples(self) -> List[Sample]:
        raise NotImplementedError
    
    def as_dataloader(self):
        raise NotImplementedError