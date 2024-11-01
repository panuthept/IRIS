from iris.data_types import Sample
from typing import List, Tuple, Callable
from easyjailbreak.models import ModelBase
from iris.augmentations.instruction_augmentations import InstructionAugmentation


class Jailbreaking(InstructionAugmentation):
    def __init__(
        self, 
        target_model: ModelBase, 
        evaluator: Callable = None,
        include_failed_cases: bool = False,
    ):
        if not include_failed_cases and evaluator is None:
            raise ValueError("Evaluator must be provided if include_failed_cases is False")
        
        self.target_model = target_model
        self.evaluator = evaluator
        self.include_failed_cases = include_failed_cases
        self.attack_success_rate = None

    def get_attack_model_name(self) -> str:
        raise NotImplementedError

    def _attack(self, instruction: str, reference_answers: List[str] = None) -> List[Tuple[str, str]]:
        """
        Return a list of tuples, each containing the jailbreak instruction and the target response.
        """
        raise NotImplementedError

    def augment(self, instruction: str, reference_answers: List[str] = None) -> List[str]:
        attack_results = self._attack(instruction=instruction, reference_answers=reference_answers)
        if not self.include_failed_cases:
            jailbreak_prompts = [jailbreak_prompt for jailbreak_prompt, response in attack_results if self.evaluator(response)]
        else:
            jailbreak_prompts = [jailbreak_prompt for jailbreak_prompt, _ in attack_results]
        return jailbreak_prompts

    def augment_batch(self, samples: List[Sample], verbose: bool = True) -> List[Sample]:
        samples = super().augment_batch(samples, verbose=verbose)

        attack_success_count = 0
        for sample in samples:
            attack_success_count += int(len(sample.instructions) > 0)
        self.attack_success_rate = attack_success_count / len(samples)
        return samples
    

class DummyJailbreaking(Jailbreaking):
    def __init__(
        self, 
        target_model: ModelBase, 
        evaluator: Callable = None,
        include_failed_cases: bool = False,
        **kwargs,
    ):
        super().__init__(
            target_model=target_model,
            evaluator=evaluator,
            include_failed_cases=include_failed_cases
        )
    
    def get_attack_model_name(self) -> str:
        return "null"

    def _attack(self, instruction: str, reference_answers: List[str] = None) -> List[Tuple[str, str]]:
        """
        Return a list of tuples, each containing the dummy jailbreak instruction (original instruction) and the target response.
        """
        target_response = self.target_model.generate(instruction)
        attack_results: List[Tuple[str, str]] = [(instruction, target_response)]
        return attack_results
    

if __name__ == "__main__":
    from iris.datasets import JailbreakBenchDataset
    from iris.model_wrappers.guard_models import WildGuard

    target_model = WildGuard(
        model_name_or_path="allenai/wildguard",
        api_key="EMPTY",
        api_base="http://10.204.100.70:11699/v1",
        cache_path="./cache",
        use_cache=True,
    )
    def evaluator(text: str) -> bool:
        return text.strip().capitalize() == "Benign"
    
    harmful_dataset = JailbreakBenchDataset(intention="harmful")
    harmful_samples = harmful_dataset.as_samples()
    
    # Test without jailbreak_template
    augmentation = DummyJailbreaking(
        target_model=target_model, 
        evaluator=evaluator,
    )

    jailbreaked_samples = augmentation.augment_batch(harmful_samples)
    print(f"ASR on Harmful prompts with DummyJailbreaking: {augmentation.attack_success_rate}")

    harmful_dataset = JailbreakBenchDataset(intention="benign")
    harmful_samples = harmful_dataset.as_samples()
    
    # Test without jailbreak_template
    augmentation = DummyJailbreaking(
        target_model=target_model, 
        evaluator=evaluator,
    )

    jailbreaked_samples = augmentation.augment_batch(harmful_samples)
    print(f"ASR on Benign prompts with DummyJailbreaking: {augmentation.attack_success_rate}")