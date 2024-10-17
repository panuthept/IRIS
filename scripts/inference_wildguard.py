import json
import argparse
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 

from tqdm import tqdm
from iris.datasets import WildGuardMixDataset
from iris.model_wrappers.guard_models import WildGuard


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack_engine", type=str, default="vanilla")
    args = parser.parse_args()

    # Initial model
    model = WildGuard(model_name_or_path="allenai/wildguard")

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