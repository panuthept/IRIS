from typing import List, TYPE_CHECKING
if TYPE_CHECKING:
    from iris.augmentations.instruction_augmentations.jailbreaks.utils.gpt_fuzzer.mutator import Mutator
    from iris.augmentations.instruction_augmentations.jailbreaks.gpt_fuzzer import GPTFuzzerJailbreaking


class PromptNode:
    def __init__(
            self,
            fuzzer: 'GPTFuzzerJailbreaking',
            prompt: str,
            responses: List[str] = None,
            results: List[int] = None,
            parent: 'PromptNode' = None,
            mutator: 'Mutator' = None
    ):
        self.fuzzer: 'GPTFuzzerJailbreaking' = fuzzer
        self.prompt: str = prompt
        self.responses: List[str] = responses if responses is not None else []
        self.results: List[int] = results if results is not None else []
        self.visited_num = 0

        self.parent: 'PromptNode' = parent
        self.mutator: 'Mutator' = mutator
        self.child: List[PromptNode] = []
        self.level: int = 0 if parent is None else parent.level + 1

        self._index: int = None

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, index: int):
        self._index = index
        if self.parent is not None:
            self.parent.child.append(self)

    @property
    def num_jailbreak(self):
        return sum(self.results)

    @property
    def num_reject(self):
        return len(self.results) - sum(self.results)

    @property
    def num_query(self):
        return len(self.results)