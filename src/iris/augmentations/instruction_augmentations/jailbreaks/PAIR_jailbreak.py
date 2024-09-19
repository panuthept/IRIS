from typing import List
from iris.augmentations.instruction_augmentations import InstructionAugmentation
from easyjailbreak.datasets.jailbreak_datasets import JailbreakDataset, Instance
from easyjailbreak.seed.seed_template import SeedTemplate
from easyjailbreak.mutation.generation import HistoricalInsight
from easyjailbreak.models import OpenaiModel, HuggingfaceModel
from easyjailbreak.metrics.Evaluator.Evaluator_GenerativeGetScore import EvaluatorGenerativeGetScore
from easyjailbreak.models import ModelBase
from iris.cache import CacheStorage
from transformers import AutoModel, AutoTokenizer


class PAIRJailbreak(InstructionAugmentation):
    def __init__(self, attack_model, target_model, eval_model, use_cache: bool = True, cache_path: str = None):
        self.attack_model = attack_model
        self.target_model = target_model
        self.eval_model = eval_model
        self.use_cache = use_cache
        self.cache_storage = CacheStorage(f"easyjailbreak/PAIR", cache_path)
        self.attack_seed, self.attack_system_message = SeedTemplate().new_seeds(method_list=["PAIR"])
        self.evaluator = EvaluatorGenerativeGetScore(self.eval_model)
        self.mutations = [HistoricalInsight(self.attack_model, attr_name=[])]
        self.n_iterations = 5
        self.max_n_attack_attempts = 5
        self.target_max_n_tokens = 150
        self.target_temperature = 1
        self.target_top_p = 0.9

    def augment(self, instruction: str, reference_response: str) -> List[str]:
        instance = Instance(query=instruction, reference_responses=[reference_response])

        if self.use_cache:
            cached_result = self.cache_storage.retrieve(instruction)
            if cached_result:
                return cached_result

        jailbreak_prompts = []
        for iteration in range(1, self.n_iterations + 1):
            for _ in range(self.max_n_attack_attempts):
                instance.jailbreak_prompt = self._generate_jailbreak_prompt(instance)
                target_response = self._generate_target_response(instance.jailbreak_prompt)
                eval_result = self.evaluator(JailbreakDataset([instance]))

                if eval_result == 10:
                    jailbreak_prompts.append(instance.jailbreak_prompt)
                    break

        if self.use_cache:
            self.cache_storage.cache(jailbreak_prompts, instruction)

        return jailbreak_prompts

    def _generate_jailbreak_prompt(self, instance: Instance) -> str:
        instance.jailbreak_prompt = self.attack_seed.format(query=instance.query, reference_responses=instance.reference_responses[0])
        self.attack_model.set_system_message(self.attack_system_message.format(query=instance.query, reference_responses=instance.reference_responses[0]))

        mutation = self.mutations[0](JailbreakDataset([instance]), prompt_format=instance.jailbreak_prompt)
        return mutation[0].jailbreak_prompt

    def _generate_target_response(self, jailbreak_prompt: str) -> str:
        if isinstance(self.target_model, OpenaiModel):
            return self.target_model.generate(jailbreak_prompt, max_tokens=self.target_max_n_tokens, temperature=self.target_temperature, top_p=self.target_top_p)
        elif isinstance(self.target_model, HuggingfaceModel):
            return self.target_model.generate(jailbreak_prompt, max_new_tokens=self.target_max_n_tokens, temperature=self.target_temperature, do_sample=True, top_p=self.target_top_p)

