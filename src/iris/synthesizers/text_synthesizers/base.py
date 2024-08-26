from typing import Any, Dict, List


class BaseTextSynthesizer:
    """Base class for text synthesizers."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def synthesize(self, query: str) -> List[str]:
        raise NotImplementedError("synthesize() method not implemented")

    def synthesize_batch(self, queries: List[str]) -> List[str]:
        return [self.synthesize(query) for query in queries]