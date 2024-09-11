from iris.augmentations.instruction_augmentations import InstructionAugmentation


class Jailbreaking(InstructionAugmentation):
    def __init__(
        self, 
        include_failed_cases: bool = False,
    ):
        self.include_failed_cases = include_failed_cases