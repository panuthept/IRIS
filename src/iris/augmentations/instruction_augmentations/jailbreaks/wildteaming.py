import math
import logging
import pandas as pd
from tqdm import tqdm
from iris.data_types import Sample
from typing import List, Dict, Callable
from easyjailbreak.models import ModelBase
from iris.augmentations.instruction_augmentations.jailbreaks import Jailbreaking
from iris.augmentations.instruction_augmentations.jailbreaks.utils.gpt_fuzzer.core import PromptNode
from iris.augmentations.instruction_augmentations.jailbreaks.utils.gpt_fuzzer.template import synthesis_message
from iris.augmentations.instruction_augmentations.jailbreaks.utils.gpt_fuzzer.selection import MCTSExploreSelectPolicy
from iris.augmentations.instruction_augmentations.jailbreaks.utils.gpt_fuzzer.mutator import (
    MutateRandomSinglePolicy, OpenAIMutatorCrossOver, OpenAIMutatorExpand,
    OpenAIMutatorGenerateSimilar, OpenAIMutatorRephrase, OpenAIMutatorShorten,
)

logger = logging.getLogger()
logger.setLevel(logging.WARNING)


class WildTeamingJailbreaking(Jailbreaking):
    def __init__(
        self, 
        target_model: ModelBase, 
        attack_model: ModelBase = None,
        evaluator: Callable = None, # A function that takes a string and returns 1 if the string is jailbroken, 0 otherwise
        max_query: int = 50000,
        max_jailbreak: int = -1,
        max_reject: int = -1,
        max_iteration: int = -1,
        energy: int = 1,
        include_failed_cases: bool = False,
        **kwargs,
    ):
        super().__init__(
            target_model=target_model,
            evaluator=evaluator,
            include_failed_cases=include_failed_cases
        )
        if attack_model is None:
            # TODO: Add a default attack model
            raise ValueError("attack_model must be provided.")
        self.attack_model = attack_model
        self.max_query = max_query
        self.max_jailbreak = max_jailbreak
        self.max_reject = max_reject
        self.max_iteration = max_iteration
        self.energy = energy
        # Prepare prompt nodes
        initial_seed = pd.read_csv("./data/datasets/prompts/GPTFuzzer.csv")["text"].to_list()
        self.prompt_nodes: List[PromptNode] = [
            PromptNode(self, prompt) for prompt in initial_seed
        ]
        self.initial_prompts_nodes = self.prompt_nodes.copy()
        for i, prompt_node in enumerate(self.prompt_nodes):
            prompt_node.index = i
        # Prepare mutate policy
        self.mutate_policy = MutateRandomSinglePolicy(
            mutators=[
                OpenAIMutatorCrossOver(attack_model, temperature=0.0),  # for reproduction only, if you want better performance, use temperature>0
                OpenAIMutatorExpand(attack_model, temperature=0.0),
                OpenAIMutatorGenerateSimilar(attack_model, temperature=0.0),
                OpenAIMutatorRephrase(attack_model, temperature=0.0),
                OpenAIMutatorShorten(attack_model, temperature=0.0)
            ],
            concatentate=True,
        )
        # Prepare select policy
        self.select_policy = MCTSExploreSelectPolicy()
        # Initial parameters
        self.current_query: int = 0
        self.current_jailbreak: int = 0
        self.current_reject: int = 0
        self.current_iteration: int = 0
        # Set fuzzer
        self.mutate_policy.fuzzer = self
        self.select_policy.fuzzer = self

    def reset(self):
        self.current_query: int = 0
        self.current_jailbreak: int = 0
        self.current_reject: int = 0
        self.current_iteration: int = 0

    def is_stop(self, attack_results: Dict[int, List[str]]) -> bool:
        # Check if the attack_results has at least one jailbreak per question, then stop
        if all(len(attack_results[idx]) > 0 for idx in attack_results):
            return True

        checks = [
            ('max_query', 'current_query'),
            ('max_jailbreak', 'current_jailbreak'),
            ('max_reject', 'current_reject'),
            ('max_iteration', 'current_iteration'),
        ]
        return any(getattr(self, max_attr) != -1 and getattr(self, curr_attr) >= getattr(self, max_attr) for max_attr, curr_attr in checks)
    
    def evaluate(
        self, 
        instructions: List[str], 
        prompt_nodes: List[PromptNode], 
        attack_results: Dict[int, List[str]], 
        verbose: bool = True
    ):
        for prompt_node in prompt_nodes:
            prompt_node.results = []
            prompt_node.responses = []
            for idx, instruction in enumerate(instructions):
                message = synthesis_message(instruction, prompt_node.prompt)
                if message is None:  # The prompt is not valid
                    prompt_node.responses = []
                    prompt_node.results = []
                    break
                response = self.target_model.generate(message)
                result = self.evaluator(response)
                prompt_node.responses.append(response)
                prompt_node.results.append(result)

                if not self.include_failed_cases and result == 1:
                    # if verbose:
                    #     # Show results only for jailbroken cases
                    #     print(f"Instruction:\n{instruction}")
                    #     print("-" * 100)
                    #     print(f"Jailbreak prompt:\n{message}")
                    #     print("=" * 100)
                    attack_results[idx].append(message)

    def update(
        self, 
        instructions: List[str], 
        prompt_nodes: List[PromptNode]
    ):
        self.current_iteration += 1
        for prompt_node in prompt_nodes:
            if prompt_node.num_jailbreak > 0:
                prompt_node.index = len(self.prompt_nodes)
                self.prompt_nodes.append(prompt_node)
            self.current_jailbreak += prompt_node.num_jailbreak
            self.current_query += prompt_node.num_query
            self.current_reject += prompt_node.num_reject
        self.select_policy.update(instructions, prompt_nodes)

    def _attack(
        self, 
        instructions: List[str], 
        reference_answers: List[str] = None, 
        verbose: bool = True
    ) -> Dict[int, List[str]]:
        self.reset()
        if self.max_iteration == -1:
            self.max_iteration = math.ceil(self.max_query / len(instructions))
        attack_results: Dict[int, List[str]] = {idx: [] for idx in range(len(instructions))}
        try:
            for _ in tqdm(range(self.max_iteration), disable=not verbose):
                seed = self.select_policy.select()
                mutated_results = self.mutate_policy.mutate_single(seed)
                self.evaluate(instructions, mutated_results, attack_results, verbose=verbose)
                self.update(instructions, mutated_results)
                # print(self.current_jailbreak, self.current_query, self.current_reject)
                if verbose:
                    print(f"query: {self.current_query}/{self.max_query}, jailbreak: {self.current_jailbreak}, asr: {sum(len(attack_results[idx]) > 0 for idx in attack_results) / len(attack_results)}\n")
                if self.is_stop(attack_results):
                    break
        except KeyboardInterrupt:
            print("Fuzzing interrupted by user!")
        return attack_results
    
    def augment(self, instruction: str, reference_answers: List[str] = None) -> List[str]:
        return self._attack([instruction])[0]
    
    def augment_sample(self, sample: Sample) -> Sample:
        original_instructions = [sample.instructions[0]]
        attack_results = self._attack(original_instructions)

        sample.instructions = attack_results[0]
        sample.reference_instruction = original_instructions[0]
        return sample

    def augment_batch(self, samples: List[Sample], verbose: bool = True) -> List[Sample]:
        original_instructions = [sample.instructions[0] for sample in samples]
        attack_results = self._attack(original_instructions, verbose=verbose)

        for idx, sample in enumerate(samples):
            sample.instructions = attack_results[idx]
            sample.reference_instruction = original_instructions[idx]

        attack_success_count = 0
        for sample in samples:
            attack_success_count += int(len(sample.instructions) > 0)
        self.attack_success_rate = attack_success_count / len(samples)
        return samples


if __name__ == "__main__":
    import os
    import json
    import random
    import numpy as np
    from llama_index.llms.openai import OpenAI
    from iris.datasets import JailbreakBenchDataset
    from iris.model_wrappers.guard_models import WildGuard, LlamaGuard
    from iris.model_wrappers.generative_models import APIGenerativeLLM

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

    attack_model=APIGenerativeLLM(
        llm=OpenAI(
            model="gpt-4o",
            api_key=os.environ.get("OPENAI_API_KEY"),
        ),
        cache_path="./cache",
        use_cache=True,
    )

    attacker = GPTFuzzerJailbreaking(
        target_model=target_model,
        attack_model=attack_model,
        evaluator=lambda x: int(x.strip().capitalize() == "Benign"),
        max_query=50000,
        # max_iteration=10,
    )

    harmful_dataset = JailbreakBenchDataset(intention="harmful")
    harmful_samples = harmful_dataset.as_samples()

    jailbreaked_samples = attacker.augment_batch(harmful_samples)
    print(f"ASR (Harmful): {attacker.attack_success_rate}")
    # Save samples
    with open(f"./jailbreaked_harmful_sample.jsonl", "w") as f:
        for sample in jailbreaked_samples:
            f.write(json.dumps(sample.as_dict()) + "\n")

    # benign_dataset = JailbreakBenchDataset(intention="benign")
    # benign_samples = benign_dataset.as_samples()

    # jailbreaked_samples = attacker.augment_batch(benign_samples)
    # print(f"ASR (Benign): {attacker.attack_success_rate}")
    # # Save samples
    # with open(f"./jailbreaked_benign_sample.jsonl", "w") as f:
    #     for sample in jailbreaked_samples:
    #         f.write(json.dumps(sample.as_dict()) + "\n")