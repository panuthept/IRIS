from typing import List
from iris.data_types import Sample, Response, Result


class BaseBenchmark:
    def get_train_set(self, *args, **kwargs) -> List[Sample]:
        raise NotImplementedError

    def get_test_set(self, *args, **kwargs) -> List[Sample]:
        raise NotImplementedError

    def evaluate(self, responses: List[Response]) -> List[Result]:
        raise NotImplementedError