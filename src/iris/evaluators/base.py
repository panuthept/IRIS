from typing import List
from iris.data_types import Sample


class BaseEvaluator:
    def evaluate(self, sample: Sample) -> Sample:
        raise NotImplementedError

    def evaluate_batch(self, samples: List[Sample]) -> List[Sample]:
        return [self.evaluate(sample) for sample in samples]