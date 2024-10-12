import os
import json
import logging
import argparse
from iris.model_wrappers.guard_models import load_guard
from iris.datasets import (
    AwesomePromptsDataset,
    JailbreakBenchDataset,
    JailbreaKV28kDataset,
    WildGuardMixDataset,
    XSTestDataset,
)

    
def get_dataset(dataset_name: str, dataset_intention: str):
    if dataset_name == "awesome_prompts":
        return AwesomePromptsDataset(intention=dataset_intention)
    elif dataset_name == "jailbreak_bench":
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
    return f"{args.artifact_dir}/{args.attacker_name}/{args.dataset_name}/{args.dataset_split}/{args.dataset_intention}.json"
    
def load(args):
    load_path = get_path(args)
    if os.path.exists(load_path):
        with open(load_path, "r") as f:
            jailbreak_artifacts = json.load(f)
            return jailbreak_artifacts
    else:
        return None


if __name__ == "__main__":
    parser = parser = argparse.ArgumentParser()
    parser.add_argument("--guard_name", type=str, default="WildGuard")
    parser.add_argument("--model_name", type=str, default="allenai/wildguard")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--api_base", type=str, default=None)
    parser.add_argument("--attacker_name", type=str, default="wildteaming")
    parser.add_argument("--artifact_dir", type=str, default="./jailbreak_artifacts")
    parser.add_argument("--dataset_name", type=str, default="jailbreak_bench")
    parser.add_argument("--dataset_intention", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="test")
    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    target_model = load_guard(args.guard_name, args.model_name, args.checkpoint_path, args.api_key, args.api_base)
    dataset = get_dataset(args.dataset_name, args.dataset_intention)
    samples = dataset.as_samples(split=args.dataset_split)

    samples = target_model.prompt_classify_batch(samples)
    harmful_samples = [sample for sample in samples if sample.instructions_true_label[0] == "Harmful"]
    benign_samples = [sample for sample in samples if sample.instructions_true_label[0] == "Benign"]
    tpr = sum([sample.instructions_pred_label[0] == "Harmful" for sample in harmful_samples]) / (len(harmful_samples) + 1e-7)
    fpr = sum([sample.instructions_pred_label[0] == "Harmful" for sample in benign_samples]) / (len(benign_samples) + 1e-7)
    print(f"TPR: {round(tpr, 2)}")
    print(f"FPR: {round(fpr, 2)}")