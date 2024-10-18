import os
import json
import random
import argparse
import numpy as np
from iris.datasets import load_dataset
from iris.augmentations.instruction_augmentations.jailbreaks import load_attacker
from iris.model_wrappers.guard_models import (
    CFIGuard,
    DummyBiasModel,
    load_guard,
)

    
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
    if sample.instructions[0] in jailbreak_artifacts:
        if args.target_model in jailbreak_artifacts[sample.instructions[0]]:
            if args.attack_model in jailbreak_artifacts[sample.instructions[0]][args.target_model]:
                return True
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="./jailbreak_artifacts")
    parser.add_argument("--save_batch", type=int, default=10)
    parser.add_argument("--ignore_existing", action="store_true")
    parser.add_argument("--guard_name", type=str, default="WildGuard")
    parser.add_argument("--model_name", type=str, default="allenai/wildguard")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--api_base", type=str, default=None)
    parser.add_argument("--counterfactual_inference", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--attacker_name", type=str, default="WildTeamingJailbreaking")
    parser.add_argument("--max_iteration", type=int, default=10)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--dataset_name", type=str, default="JailbreakBenchDataset")
    parser.add_argument("--dataset_intention", type=str, default="harmful")
    parser.add_argument("--dataset_split", type=str, default="test")
    args = parser.parse_args()

    # Set random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    target_model = load_guard(args.guard_name, args.model_name, args.checkpoint_path, args.api_key, args.api_base)
    if args.counterfactual_inference:
        target_model = CFIGuard(
            target_model=target_model,
            bias_model=DummyBiasModel(),
            alpha=args.alpha,
        )
    evaluator = lambda x: int(x.strip().capitalize() == "Benign")
    attacker = load_attacker(args.attacker_name, target_model, evaluator, args.max_iteration)
    dataset = load_dataset(args.dataset_name, args.dataset_intention)
    samples = dataset.as_samples(split=args.dataset_split)
    args.attack_model = attacker.get_attack_model_name()
    args.target_model = target_model.get_model_name()
    print(args.target_model)

    # Load jailbreak artifacts, if exists
    jailbreak_artifacts = load(args)
    if jailbreak_artifacts is not None:
        jailbreak_artifacts = jailbreak_artifacts["jailbreak_artifacts"]

    if args.attacker_name == "gpt_fuzzer":
        if any([is_skip(args, sample, jailbreak_artifacts) for sample in samples]):
            print("All samples are already processed. Skipping...")
        else:
            # GPTFUZZER is designed to process samples all at once
            attacked_samples = attacker.augment_batch(samples, verbose=True)
            # Save jailbreak artifacts
            save(args, attacked_samples)
    else:
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