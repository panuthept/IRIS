import os
import json
import random
import argparse
import numpy as np
from iris.model_wrappers.guard_models import (
    GuardLLM,
    WildGuard,
    LlamaGuard,
    ShieldGemma
)
from iris.augmentations.instruction_augmentations.jailbreaks import (
    GPTFuzzerJailbreaking,
    WildTeamingJailbreaking,
    MultiLingualJailbreaking,
)
from iris.datasets import (
    JailbreakBenchDataset,
    JailbreaKV28kDataset,
    WildGuardMixDataset,
    XSTestDataset,
)


def get_model(target_model: str, api_key: str, api_base: str):
    if "wildguard" in target_model:
        return WildGuard(
            model_name_or_path=target_model,
            api_key=api_key,
            api_base=api_base,
        )
    elif "llama" in target_model:
        return LlamaGuard(
            model_name_or_path=target_model,
            api_key=api_key,
            api_base=api_base,
        )
    elif "google" in target_model:
        return ShieldGemma(
            model_name_or_path=target_model,
            api_key=api_key,
            api_base=api_base,
        )
    else:
        raise ValueError(f"Invalid target model: {target_model}")
    
def get_attacker(attacker_name: str, target_model: GuardLLM, max_iteration: int = 10):
    evaluator = lambda x: int(x.strip().capitalize() == "Benign")
    if attacker_name == "gpt_fuzzer":
        attacker = GPTFuzzerJailbreaking(
            target_model=target_model,
            evaluator=evaluator,
            max_iteration=max_iteration,
        )
        attack_model = attacker.attack_model.get_model_name()
    elif attacker_name == "wildteaming":
        attacker = WildTeamingJailbreaking(
            target_model=target_model,
            evaluator=evaluator,
            max_iteration=max_iteration,
        )
        attack_model = attacker.attack_model.get_model_name()
    elif attacker_name == "multilingual":
        attacker = MultiLingualJailbreaking(
            target_model=target_model,
            evaluator=evaluator,
        )
        attack_model = "google_translator_api"
    elif attacker_name == "multilingual+":
        attacker = MultiLingualJailbreaking(
            target_model=target_model,
            evaluator=evaluator,
            apply_jailbreak_template=True,
        )
        attack_model = "google_translator_api"
    else:
        raise ValueError(f"Invalid attacker name: {attacker_name}")
    return attacker, attack_model
    
def get_dataset(dataset_name: str, dataset_intention: str):
    if dataset_name == "jailbreak_bench":
        return JailbreakBenchDataset(intention=dataset_intention)
    elif dataset_name == "jailbreak_kv28k":
        assert dataset_intention != "harmful", "JailbreakKV28k only has harmful intention"
        return JailbreaKV28kDataset(intention=dataset_intention)
    elif dataset_name == "wildguard_mix":
        return WildGuardMixDataset(intention=dataset_intention)
    elif dataset_name == "xs_test":
        return XSTestDataset(intention=dataset_intention)
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")
    
def get_path(args):
    return f"{args.save_dir}/{args.attacker_name}/{args.dataset_name}/{args.dataset_split}/{args.dataset_intention}.json"
    
def load(args):
    load_path = get_path(args)
    if os.path.exists(load_path):
        with open(load_path, "r") as f:
            jailbreak_artifacts = json.load(f)
            return jailbreak_artifacts
    else:
        return None
    
def save(args, attacked_samples):
    save_path = get_path(args)
    # Load jailbreak artifacts, if exists
    jailbreak_artifacts = load(args)
    if jailbreak_artifacts is None:
        jailbreak_artifacts = {
            "dataset": args.dataset_name,
            "attacker": args.attacker_name,
            "max_iteration": args.max_iteration,
            "jailbreak_artifacts": {},
        }
    # Update jailbreak artifacts
    for sample in attacked_samples:
        if sample.reference_instruction not in jailbreak_artifacts["jailbreak_artifacts"]:
            jailbreak_artifacts["jailbreak_artifacts"][sample.reference_instruction] = {}
        if args.target_model not in jailbreak_artifacts["jailbreak_artifacts"][sample.reference_instruction]:
            jailbreak_artifacts["jailbreak_artifacts"][sample.reference_instruction][args.target_model] = {}
        if args.attack_model not in jailbreak_artifacts["jailbreak_artifacts"][sample.reference_instruction][args.target_model]:
            jailbreak_artifacts["jailbreak_artifacts"][sample.reference_instruction][args.target_model][args.attack_model] = []
        jailbreak_artifacts["jailbreak_artifacts"][sample.reference_instruction][args.target_model][args.attack_model].extend(sample.instructions)

    # Save jailbreak artifacts
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        f.write(json.dumps(jailbreak_artifacts, indent=4))

def is_skip(args, sample, jailbreak_artifacts):
    if args.ignore_existing:
        return False
    if jailbreak_artifacts is None:
        return False
    if sample.reference_instruction in jailbreak_artifacts:
        if args.target_model in jailbreak_artifacts[sample.reference_instruction]:
            if args.attack_model in jailbreak_artifacts[sample.reference_instruction][args.target_model]:
                return True
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="./jailbreak_artifacts")
    parser.add_argument("--save_batch", type=int, default=10)
    parser.add_argument("--ignore_existing", action="store_true")
    parser.add_argument("--target_model", type=str, default="allenai/wildguard")
    parser.add_argument("--api_key", type=str, default="EMPTY")
    parser.add_argument("--api_base", type=str, default="http://10.204.100.70:11699/v1")
    parser.add_argument("--attacker_name", type=str, default="wildteaming")
    parser.add_argument("--max_iteration", type=int, default=10)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--dataset_name", type=str, default="jailbreak_bench")
    parser.add_argument("--dataset_intention", type=str, default="harmful")
    parser.add_argument("--dataset_split", type=str, default="test")
    args = parser.parse_args()

    # Set random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    target_model = get_model(args.target_model, args.api_key, args.api_base)
    attacker, attack_model = get_attacker(args.attacker_name, target_model, args.max_iteration)
    dataset = get_dataset(args.dataset_name, args.dataset_intention)
    samples = dataset.as_samples(split=args.dataset_split)
    args.attack_model = attack_model

    # Load jailbreak artifacts, if exists
    jailbreak_artifacts = load(args)

    # Process samples in batch
    for i in range(0, len(samples), args.save_batch):
        print(f"Processing batch {i} - {i+args.save_batch}")
        batch = [sample for sample in samples[i:i+args.save_batch] if not is_skip(args, sample, jailbreak_artifacts)]
        if len(batch) == 0:
            # Skip if all samples are already processed
            print("All samples are already processed. Skipping...")
            continue
        # Generate jailbreak artifacts
        attacked_samples = attacker.augment_batch(batch, verbose=True)
        # Save jailbreak artifacts
        save(args, attacked_samples)