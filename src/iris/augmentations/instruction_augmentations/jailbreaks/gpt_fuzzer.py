import logging
import pandas as pd
from tqdm import tqdm
from typing import List, Tuple, Callable
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


class GPTFuzzerJailbreaking(Jailbreaking):
    def __init__(
        self, 
        target_model: ModelBase, 
        attack_model: ModelBase = None,
        evaluator: Callable = None, # A function that takes a string and returns 1 if the string is jailbroken, 0 otherwise
        max_query: int = -1,
        max_jailbreak: int = 1,
        max_reject: int = -1,
        max_iteration: int = 10,
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
        self.mutate_policy = MutateRandomSinglePolicy([
            OpenAIMutatorCrossOver(attack_model, temperature=0.0),  # for reproduction only, if you want better performance, use temperature>0
            OpenAIMutatorExpand(attack_model, temperature=0.0),
            OpenAIMutatorGenerateSimilar(attack_model, temperature=0.0),
            OpenAIMutatorRephrase(attack_model, temperature=0.0),
            OpenAIMutatorShorten(attack_model, temperature=0.0)],
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

    def is_stop(self):
        checks = [
            ('max_query', 'current_query'),
            ('max_jailbreak', 'current_jailbreak'),
            ('max_reject', 'current_reject'),
            ('max_iteration', 'current_iteration'),
        ]
        return any(getattr(self, max_attr) != -1 and getattr(self, curr_attr) >= getattr(self, max_attr) for max_attr, curr_attr in checks)
    
    def evaluate(self, instruction: str, prompt_nodes: List[PromptNode], attack_results: List[Tuple[str, str]]):
        for prompt_node in prompt_nodes:
            message = synthesis_message(instruction, prompt_node.prompt)
            if message is None:  # The prompt is not valid
                prompt_node.responses = []
                prompt_node.results = []
                break
            print("-" * 100)
            print(message)
            print("-" * 100)
            response = self.target_model.generate(message)
            print(response)
            eval_result = self.evaluator(response)
            print(eval_result)
            print("*" * 100)
            prompt_node.responses = [response]
            prompt_node.results = [eval_result]
            attack_results.append((message, response))

    def update(self, instruction: str, prompt_nodes: List[PromptNode]):
        self.current_iteration += 1
        for prompt_node in prompt_nodes:
            if prompt_node.num_jailbreak > 0:
                prompt_node.index = len(self.prompt_nodes)
                self.prompt_nodes.append(prompt_node)
            self.current_jailbreak += prompt_node.num_jailbreak
            self.current_query += prompt_node.num_query
            self.current_reject += prompt_node.num_reject
        self.select_policy.update([instruction], prompt_nodes)

    def _attack(self, instruction: str, reference_answers: List[str] = None) -> List[Tuple[str, str]]:
        attack_results = []
        try:
            for _ in tqdm(range(self.max_iteration)):
                seed = self.select_policy.select()
                mutated_results = self.mutate_policy.mutate_single(seed)
                self.evaluate(instruction, mutated_results, attack_results)
                self.update(instruction, mutated_results)
                print(self.current_jailbreak, self.current_query, self.current_reject)
                if self.is_stop():
                    break
        except KeyboardInterrupt:
            print("Fuzzing interrupted by user!")
        return attack_results


if __name__ == "__main__":
    import os
    from iris.datasets import JailbreakBenchDataset
    from llama_index.llms.openai_like import OpenAILike
    from iris.model_wrappers.guard_models import WildGuard, LlamaGuard
    from iris.model_wrappers.generative_models import APIGenerativeLLM

    # target_model = WildGuard(
    #     model_name_or_path="allenai/wildguard",
    #     api_key="EMPTY",
    #     api_base="http://10.204.100.70:11699/v1",
    #     cache_path="./cache",
    #     use_cache=True,
    # )
    target_model = LlamaGuard(
        model_name_or_path="meta-llama/Llama-Guard-3-8B",
        api_key="EMPTY",
        api_base="http://10.204.100.70:11700/v1",
        cache_path="./cache",
        use_cache=True,
    )

    attack_model=APIGenerativeLLM(
        llm=OpenAILike(
            model="gpt-3.5-turbo-instruct",
            api_key=os.environ.get("OPENAI_API_KEY"),
        ),
        cache_path="./cache",
        use_cache=False,
    )

    attacker = GPTFuzzerJailbreaking(
        target_model=target_model,
        attack_model=attack_model,
        evaluator=lambda x: int(x.strip().capitalize() == "Benign"),
    )

    harmful_dataset = JailbreakBenchDataset(intention="harmful")
    harmful_samples = harmful_dataset.as_samples()

    # jailbreaked_samples = attacker.augment_batch(harmful_samples)
    jailbreaked_sample = attacker.augment_sample(harmful_samples[0])
    print(jailbreaked_sample)

    benign_dataset = JailbreakBenchDataset(intention="benign")
    benign_samples = benign_dataset.as_samples()

    # jailbreaked_samples = attacker.augment_batch(harmful_samples)
    jailbreaked_sample = attacker.augment_sample(benign_samples[0])
    print(jailbreaked_sample)