import random
import json
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


class DeepInceptionJailbreaking(Jailbreaking):
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

    def seed_template(self, seeds_num: int = 1, prompt_usage: str = "attack", methods: List[str] = None, template_file: str = None):
        seeds = []
        if methods is None:
            methods = ["default"]
        
        template_dict = json.load(open(template_file, "r", encoding="utf-8"))

        template_pool = []
        for method in methods:
            try:
                template_pool.extend(template_dict[prompt_usage][method])
            except KeyError:
                raise AttributeError("{} contains no {} prompt template from the method {}".
                                    format(template_file, prompt_usage, method))

        if seeds_num is None:
            return template_pool
        else:
            assert seeds_num > 0, "The seeds_num must be a positive integer."
            assert seeds_num <= len(template_pool), \
            "The number of seeds in the template pool is les than the number being asked for."
            index_list = random.sample(range(len(template_pool)), seeds_num)
            for index in index_list:
                seeds.append(template_pool[index])
                
        return seeds

    def _attack(self, instruction: str, reference_answers: List[str] = None) -> List[Tuple[str, str]]:
        """
        Return a list of tuples, each containing the deepinception jailbreak instruction (original instruction) and the target response.
        """
        template_file = "./data/datasets/easyjailbreak/easyjailbreak/seed/seed_template.json"
        init_prompt = self.seed_template(seeds_num=1, prompt_usage='attack', methods=['DeepInception'], template_file=template_file)
        prompt = init_prompt[0]
        
        target_response = self.target_model.generate(prompt.format(query=instruction))
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
    attacker = DeepInceptionJailbreaking(
        target_model=target_model, 
        evaluator=evaluator,
    )

    jailbreaked_samples = attacker.augment_batch(harmful_samples)
    print(f"ASR on Harmful prompts with DeepInceptionJailbreaking: {attacker.attack_success_rate}")

    harmful_dataset = JailbreakBenchDataset(intention="benign")
    harmful_samples = harmful_dataset.as_samples()
    
    # Test without jailbreak_template
    attacker = DeepInceptionJailbreaking(
        target_model=target_model, 
        evaluator=evaluator,
    )

    jailbreaked_samples = attacker.augment_batch(harmful_samples)
    print(f"ASR on Benign prompts with DeepInceptionJailbreaking: {attacker.attack_success_rate}")