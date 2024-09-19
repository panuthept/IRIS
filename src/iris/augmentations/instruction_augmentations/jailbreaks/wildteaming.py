import os
import logging
from iris.cache import CacheMode
from easyjailbreak.models import ModelBase
from typing import List, Callable, Optional
from llama_index.llms.together import TogetherLLM
from llama_index.llms.openai_like import OpenAILike
from iris.model_wrappers.generative_models import APIGenerativeLLM
from iris.augmentations.instruction_augmentations.jailbreaks import Jailbreaking
from iris.augmentations.instruction_augmentations.jailbreaks.utils.wildteaming.eval_utils import get_pruner

logger = logging.getLogger()
logger.setLevel(logging.WARNING)


class WildTeamingJailbreaking(Jailbreaking):
    def __init__(
        self, 
        target_model: ModelBase, 
        evaluator: Callable,    # A function that takes a string and returns 1 if the string is jailbroken, 0 otherwise
        attack_model: Optional[ModelBase] = None, 
        attacker_type: str = "fix_lead_seed_sentence",
        num_tactics_per_attack: int = 3,
        max_iteration: int = 50,
        max_jailbreak: int = 1,
        include_failed_cases: bool = False, # Include failed cases in the final results?
        **kwargs,
    ):  
        if attack_model is None:
            attack_model = APIGenerativeLLM(
                llm=TogetherLLM(
                    # model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                    # api_key=os.environ.get("TOGETHERAI_API_KEY"),
                    model="TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ",
                    api_key="EMPTY",
                    api_base="http://10.204.100.79:11700/v1",
                ),
                cache_path="./cache",
                use_cache=True,
                cache_mode=CacheMode.NO_DUPLICATE,
            )
        self.attack_model = attack_model
        attacker_config = {
            "attacker_type": attacker_type,
            "num_tokens": 1024,
            "temperature": 1,
            "top_p": 0.9,
            "num_tactics_per_attack": num_tactics_per_attack,
            "num_excerpts_per_tactic": 0,
            "tactics_selection_method": "random_common_prioritized",
        }
        self.attacker = Attacker(attacker_config=attacker_config, model=self.attack_model)
        self.attacker_type = attacker_type
        self.num_tactics_per_attack = num_tactics_per_attack
        self.max_iteration = max_iteration
        self.max_jailbreak = max_jailbreak
        self.off_topics_pruner = get_pruner("wanli")

        super().__init__(
            target_model=target_model,
            evaluator=evaluator,
            include_failed_cases=include_failed_cases
        )
    
    def _attack(self, instruction: str, num_attacks: int, reference_answers: List[str] = None) -> List[str]:
        if reference_answers is not None:
            reference_answer = reference_answers[0]
        # batch_behavior_attacks = self.attacker.get_attacks(instruction, num_attacks, reference_answer)[1]
        all_attacks_gen_prompts, all_tactics = self.attacker._format_all_attacks_gen_prompts(instruction, num_attacks, reference_answer)
        # print(all_attacks_gen_prompts)
        responses = [self.attack_model.generate(
            prompt=all_attacks_gen_prompt,
            temperature=1.0,
            max_tokens=1024,
            top_p=0.9, 
            ) for all_attacks_gen_prompt in all_attacks_gen_prompts
        ]
        # print("-" * 100)
        # print(responses)
        # print("=" * 100)
        all_attacks = self.attacker._parse_attacks(responses)
        return all_attacks

    def _evaluate(self, instruction: str, response: str, reference_answers: List[str] = None) -> bool:
        off_topics_prune_label = self.off_topics_pruner.prune_off_topics(instruction, [response])[0][0]
        low_risk_prune_label = self.evaluator(response)
        return off_topics_prune_label == 0 and low_risk_prune_label == 1

    def augment(self, instruction: str, reference_answers: List[str] = None) -> List[str]:
        attack_prompts: List[str] = []
        jailbreak_prompts: List[str] = []
        for _ in range(self.max_iteration):
            num_attacks = max(self.max_jailbreak - len(jailbreak_prompts), 0)
            attack_prompts: List[str] = self._attack(instruction, num_attacks, reference_answers)
            responses = [self.target_model.generate(attack_prompt) for attack_prompt in attack_prompts]
            results = [self._evaluate(instruction, response, reference_answers) for response in responses]
            jailbreak_prompts.extend([attack_prompt for attack_prompt, result in zip(attack_prompts, results) if result])
            attack_prompts.extend(attack_prompts)
            # Stop condition
            if len(jailbreak_prompts) >= self.max_jailbreak:
                break

        if self.include_failed_cases:
            return attack_prompts
        else:
            return jailbreak_prompts


if __name__ == "__main__":
    import json
    import random
    import numpy as np
    from tqdm import tqdm
    from iris.datasets import JailbreakBenchDataset
    from iris.model_wrappers.guard_models import WildGuard, LlamaGuard
    from iris.augmentations.instruction_augmentations.jailbreaks.utils.wildteaming.Attacker import Attacker
    from iris.augmentations.instruction_augmentations.jailbreaks.utils.wildteaming.eval_utils import get_pruner

    random.seed(42)
    np.random.seed(42)

    target_model = WildGuard(
        model_name_or_path="allenai/wildguard",
        api_key="EMPTY",
        api_base="http://10.204.100.70:11699/v1",
        cache_path="./cache",
        use_cache=True,
    )
    # target_model = LlamaGuard(
    #     model_name_or_path="meta-llama/Llama-Guard-3-8B",
    #     api_key="EMPTY",
    #     api_base="http://10.204.100.70:11700/v1",
    #     cache_path="./cache",
    #     use_cache=True,
    # )

    attacker = WildTeamingJailbreaking(
        target_model=target_model,
        evaluator=lambda x: int(x.strip().capitalize() == "Benign"),
        max_iteration=5,
        max_jailbreak=1,
    )

    harmful_dataset = JailbreakBenchDataset(intention="harmful")
    harmful_samples = harmful_dataset.as_samples()

    jailbreaked_samples = attacker.augment_batch(harmful_samples)
    print(f"ASR (Harmful): {attacker.attack_success_rate}")
    # Save samples
    with open(f"./jailbreaked_harmful_sample.jsonl", "w") as f:
        for sample in jailbreaked_samples:
            f.write(json.dumps(sample.as_dict()) + "\n")

    benign_dataset = JailbreakBenchDataset(intention="benign")
    benign_samples = benign_dataset.as_samples()

    jailbreaked_samples = attacker.augment_batch(benign_samples)
    print(f"FPR (Benign): {1 - attacker.attack_success_rate}")
    # Save samples
    with open(f"./jailbreaked_benign_sample.jsonl", "w") as f:
        for sample in jailbreaked_samples:
            f.write(json.dumps(sample.as_dict()) + "\n")