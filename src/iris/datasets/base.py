from typing import List, Dict
from iris.data_types import Sample
from abc import ABC, abstractmethod


class Dataset(ABC):
    @abstractmethod
    def _load_dataset(self) -> List[Dict]:
        raise NotImplementedError

    def as_samples(self) -> List[Sample]:
        raise NotImplementedError
    
    def as_dataloader(self):
        raise NotImplementedError