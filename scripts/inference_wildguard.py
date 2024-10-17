import json
import argparse
from tqdm import tqdm
from iris.datasets import WildGuardMixDataset
from iris.model_wrappers.guard_models import WildGuard


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--checkpoint_path", type=str, default="./finetuned_models/sft_wildguard/checkpoint-1220")
    parser.add_argument("--attack_engine", type=str, default="vanilla")
    args = parser.parse_args()

    # Initial model
    model = WildGuard(
        model_name_or_path=args.model_name,
        checkpoint_path=args.checkpoint_path,
    )

    # Harmful prompts
    dataset = WildGuardMixDataset(intention="harmful", attack_engine=args.attack_engine)
    samples = dataset.as_samples(split="test")

    for sample in tqdm(samples):
        prompts = sample.get_prompts()
        labels = ["Harmful"] * len(prompts)
        for prompt, label in zip(prompts, labels):
            response = model.generate(prompt, return_probs=False)
            activations = model.model.logitlens.get_last_activations()
            with open("./harmful_prompts.jsonl", "a") as f:
                f.write(json.dumps({
                    "prompt": prompt,
                    "response": response,
                    "label": label,
                    "activations": activations
                }, ensure_ascii=False) + "\n")

    # Benign prompts
    dataset = WildGuardMixDataset(intention="benign", attack_engine=args.attack_engine)
    samples = dataset.as_samples(split="test")

    for sample in tqdm(samples):
        prompts = sample.get_prompts()
        labels = ["Benign"] * len(prompts)
        for prompt, label in zip(prompts, labels):
            response = model.generate(prompt, return_probs=False)
            activations = model.model.logitlens.get_last_activations()
            with open("./benign_prompts.jsonl", "a") as f:
                f.write(json.dumps({
                    "prompt": prompt,
                    "response": response,
                    "label": label,
                    "activations": activations
                }, ensure_ascii=False) + "\n")