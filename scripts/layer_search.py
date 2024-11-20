import os
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from iris.datasets import WildGuardMixDataset
from iris.model_wrappers.guard_models import WildGuard


"""
CUDA_VISIBLE_DEVICES=0 python scripts/layer_search.py \
--model_name allenai/wildguard \
--save_logits \
--save_path ./layer_search_outputs/wildguard/logits.jsonl

CUDA_VISIBLE_DEVICES=0 python scripts/layer_search.py \
--model_name allenai/wildguard \
--save_activations \
--save_path ./layer_search_outputs/wildguard/activations.jsonl

CUDA_VISIBLE_DEVICES=0 python scripts/layer_search.py \
--model_name allenai/wildguard \
--save_logits \
--save_activations \
--save_path ./layer_search_outputs/wildguard/activations_and_logits.jsonl

CUDA_VISIBLE_DEVICES=1 python scripts/layer_search.py \
--model_name allenai/wildguard \
--checkpoint_path ./finetuned_models/iris_wildguard_layer_19/checkpoint-1220 \
--save_logits \
--save_path ./layer_search_outputs/iris_wildguard_layer_19/logits.jsonl

CUDA_VISIBLE_DEVICES=1 python scripts/layer_search.py \
--model_name allenai/wildguard \
--checkpoint_path ./finetuned_models/iris_wildguard_layer_19/checkpoint-1220 \
--save_activations \
--save_path ./layer_search_outputs/iris_wildguard_layer_19/activations.jsonl
"""


def distance_function(activations, centroids, distance_function):
    if distance_function == "L2":
        return np.linalg.norm(activations - centroids, axis=-1)
    elif distance_function == "Cosine":
        return 1 - np.matmul(activations, centroids.T).squeeze() / (np.linalg.norm(activations, axis=-1) * np.linalg.norm(centroids, axis=-1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="allenai/wildguard")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--train_eval_split", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--distance_function", type=str, default="L2", choices=["L2", "Cosine"])
    parser.add_argument("--max_benign", type=int, default=1000)
    parser.add_argument("--max_harmful", type=int, default=1000)
    parser.add_argument("--save_logits", action="store_true")
    parser.add_argument("--save_activations", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--save_path", type=str, default="./activations.jsonl")
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        random.seed(args.seed)

        # Load model
        model = WildGuard(
            model_name_or_path=args.model_name,
            checkpoint_path=args.checkpoint_path,
        )
        
        # Load dataset
        dataset = WildGuardMixDataset(attack_engine="vanilla")
        samples = dataset.as_samples(split="train")

        # Get development set
        random.shuffle(samples)
        if args.train_eval_split == 1.0:
            train_samples, eval_samples = samples, []
        else:
            train_size = int(len(samples) * args.train_eval_split)
            train_samples, eval_samples = samples[:train_size], samples[train_size:]
        print(f"Dev size: {len(eval_samples)}")

        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

        benign_count = 0
        harmful_count = 0
        with open(args.save_path, "w") as f:
            for sample in tqdm(eval_samples):
                prompts = sample.get_prompts()
                gold_labels = sample.instructions_true_label
                for prompt, gold_label in zip(prompts, gold_labels):
                    response = model.generate(prompt, return_probs=True)
                    classified_label = sorted(response, key=lambda x: x[1], reverse=True)[0][0]
                    if classified_label != gold_label:
                        continue
                    if gold_label == "Harmful" and harmful_count > args.max_harmful:
                        continue
                    if gold_label == "Benign" and benign_count > args.max_benign:
                        continue
                    if gold_label == "Harmful":
                        harmful_count += 1
                    else:
                        benign_count += 1
                    cache = model.model.logitlens.fetch_cache(
                        return_tokens=False, 
                        return_logits=args.save_logits, 
                        return_activations=args.save_activations,
                    )
                    # Remove self_attn and mlp
                    cache = {k: {module_name: activation for module_name, activation in v.items() if "self_attn" not in module_name and "mlp" not in module_name} for k, v in cache.items()}
                    f.write(json.dumps({
                        "prompt": prompt,
                        "response": response,
                        "label": gold_label,
                        "cache": cache
                    }, ensure_ascii=False) + "\n")
                    if benign_count > args.max_benign and harmful_count > args.max_harmful:
                        break
                if benign_count > args.max_benign and harmful_count > args.max_harmful:
                    break
    if args.plot:
        responses = []
        with open(args.save_path, "r") as f:
            for line in tqdm(open(args.save_path)):
                data = json.loads(line)
                responses.append(data)
        layer_names = list(responses[0]["cache"]["activations"].keys())
        print(layer_names)

        # Get mean and std
        activations = {
            "Harmful": {layer_name: [] for layer_name in layer_names}, 
            "Benign": {layer_name: [] for layer_name in layer_names},
        }
        centroids = {
            "Harmful": {layer_name: [] for layer_name in layer_names},
            "Benign": {layer_name: [] for layer_name in layer_names},
            "Any": {layer_name: [] for layer_name in layer_names},
        }
        distances = {
            "Harmful_Harmful": {layer_name: [] for layer_name in layer_names},
            "Harmful_Benign": {layer_name: [] for layer_name in layer_names}, 
            "Benign_Benign": {layer_name: [] for layer_name in layer_names}, 
            "Benign_Harmful": {layer_name: [] for layer_name in layer_names}, 
            "Any": {layer_name: [] for layer_name in layer_names},
        }
        for layer_name in layer_names:
            for response in responses:
                label = response["label"]
                activation = response["cache"]["activations"][layer_name][0]
                activations[label][layer_name].append(activation)
            # Convert to numpy array
            activations["Harmful"][layer_name] = np.array(activations["Harmful"][layer_name])
            activations["Benign"][layer_name] = np.array(activations["Benign"][layer_name])
            # Get centroids
            centroids["Harmful"][layer_name] = activations["Harmful"][layer_name].mean(axis=0, keepdims=True)
            centroids["Benign"][layer_name] = activations["Benign"][layer_name].mean(axis=0, keepdims=True)
            centroids["Any"][layer_name] = np.concatenate([activations["Harmful"][layer_name], activations["Benign"][layer_name]]).mean(axis=0, keepdims=True)
            # Get distances
            distances["Harmful_Harmful"][layer_name] = distance_function(activations["Harmful"][layer_name], centroids["Harmful"][layer_name], args.distance_function)
            distances["Harmful_Benign"][layer_name] = distance_function(activations["Harmful"][layer_name], centroids["Benign"][layer_name], args.distance_function)
            distances["Benign_Benign"][layer_name] = distance_function(activations["Benign"][layer_name], centroids["Benign"][layer_name], args.distance_function)
            distances["Benign_Harmful"][layer_name] = distance_function(activations["Benign"][layer_name], centroids["Harmful"][layer_name], args.distance_function)
            distances["Any"][layer_name] = distance_function(np.concatenate([activations["Harmful"][layer_name], activations["Benign"][layer_name]]), centroids["Any"][layer_name], args.distance_function)

        colors = {
            "Harmful_Harmful": "red",
            "Harmful_Benign": "orange",
            "Benign_Benign": "green",
            "Benign_Harmful": "blue",
        }
        variations = {
            "Harmful_Harmful": None,
            "Harmful_Benign": None,
            "Benign_Benign": None,
            "Benign_Harmful": None,
        }

        # Plot (x-axis: layer_name, y-axis: variations)
        xs = list(range(len(layer_names)))
        for name in variations:
            variations[name] = np.array([distances[name][layer_name].mean() / distances["Any"][layer_name].mean() for layer_name in layer_names])
            plt.plot(xs, variations[name], label=name, color=colors[name])
        plt.plot(xs, [1.0] * len(xs), label="Any", linestyle="--", color="black")
        plt.xlabel("Layer")
        plt.ylabel("Normalized Variation")
        plt.legend()
        plt.show()

        diffs = {
            "Harmful": variations["Harmful_Harmful"] - variations["Harmful_Benign"],
            "Benign": variations["Benign_Benign"] - variations["Benign_Harmful"],
        }
        for label in diffs:
            plt.plot(xs, diffs[label], label=label)
        plt.xlabel("Layer")
        plt.ylabel("Normalized Variation")
        plt.legend()
        plt.show()