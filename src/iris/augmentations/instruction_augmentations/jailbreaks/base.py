from typing import List
from iris.data_types import Sample
from iris.augmentations.instruction_augmentations import InstructionAugmentation


class Jailbreaking(InstructionAugmentation):
    def __init__(
        self, 
        include_failed_cases: bool = False,
    ):
        self.include_failed_cases = include_failed_cases
        self.attack_success_rate = None

    def augment_batch(self, samples: List[Sample], verbose: bool = True) -> List[Sample]:
        samples = super().augment_batch(samples, verbose=verbose)

        attack_success_count = 0
        for sample in samples:
            attack_success_count += int(len(sample.instructions) > 0)
        self.attack_success_rate = attack_success_count / len(samples)
        return samples