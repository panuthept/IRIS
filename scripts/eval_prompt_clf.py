import os
import logging
import argparse
from iris.model_wrappers.guard_models import (
    WildGuard,
    LlamaGuard,
    ShieldGemma,
    CFIGuard,
    DummyBiasModel,
)
from iris.datasets import (
    AwesomePromptsDataset,
    JailbreakBenchDataset,
    JailbreaKV28kDataset,
    WildGuardMixDataset,
    XSTestDataset,
)

def get_model(target_model: str, api_key: str, api_base: str, counterfactual_inference: bool = False, alpha: float = 0.9):
    if "wildguard" in target_model:
        model = WildGuard(
            model_name_or_path=target_model,
            api_key=api_key,
            api_base=api_base,
        )
    elif "llama" in target_model:
        model = LlamaGuard(
            model_name_or_path=target_model,
            api_key=api_key,
            api_base=api_base,
        )
    elif "google" in target_model:
        model = ShieldGemma(
            model_name_or_path=target_model,
            api_key=api_key,
            api_base=api_base,
        )
    else:
        raise ValueError(f"Invalid target model: {target_model}")
    
    if counterfactual_inference:
        model = CFIGuard(
            target_model=model,
            bias_model=DummyBiasModel(),
            alpha=alpha,
        )
    return model
    
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


if __name__ == "__main__":
    parser = parser = argparse.ArgumentParser()
    parser.add_argument("--target_model", type=str, default="allenai/wildguard")
    parser.add_argument("--api_key", type=str, default="EMPTY")
    parser.add_argument("--api_base", type=str, default="http://10.204.100.70:11699/v1")
    parser.add_argument("--counterfactual_inference", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--dataset_name", type=str, default="jailbreak_bench")
    parser.add_argument("--dataset_intention", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="test")
    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    target_model = get_model(args.target_model, args.api_key, args.api_base, args.counterfactual_inference, args.alpha)
    dataset = get_dataset(args.dataset_name, args.dataset_intention)
    samples = dataset.as_samples(split=args.dataset_split)

    samples = target_model.prompt_classify_batch(samples)
    harmful_samples = [sample for sample in samples if sample.instructions_true_label[0] == "Harmful"]
    benign_samples = [sample for sample in samples if sample.instructions_true_label[0] == "Benign"]
    tpr = sum([sample.instructions_pred_label[0] == "Harmful" for sample in harmful_samples]) / (len(harmful_samples) + 1e-7)
    fpr = sum([sample.instructions_pred_label[0] == "Harmful" for sample in benign_samples]) / (len(benign_samples) + 1e-7)
    print(f"TPR: {round(tpr, 2)}")
    print(f"FPR: {round(fpr, 2)}")