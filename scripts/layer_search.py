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
        return np.linalg.norm(activations - centroids)
    elif distance_function == "Dot":
        return np.dot(activations, centroids)
    elif distance_function == "Cosine":
        return np.dot(activations, centroids) / (np.linalg.norm(activations) * np.linalg.norm(centroids))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="allenai/wildguard")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--train_eval_split", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--distance_function", type=str, default="L2", choices=["L2", "Dot", "Cosine"])
    parser.add_argument("--max_benign", type=int, default=1000)
    parser.add_argument("--max_harmful", type=int, default=1000)
    parser.add_argument("--save_logits", action="store_true")
    parser.add_argument("--save_activations", action="store_true")
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

    responses = []
    with open(args.save_path, "r") as f:
        for line in tqdm(open(args.save_path)):
            data = json.loads(line)
            responses.append(data)
    layer_names = list(responses[0]["cache"]["activations"].keys())
    print(layer_names)

    # Get centroids
    centroids = {
        "Harmful": {layer_name: [] for layer_name in layer_names}, 
        "Benign": {layer_name: [] for layer_name in layer_names},
    }
    for layer_name in layer_names:
        for response in responses:
            label = response["label"]
            activation = response["cache"]["activations"][layer_name][0]
            centroids[label][layer_name].append(activation)
        centroids[label][layer_name] = np.array(centroids[label][layer_name]).mean(axis=0)

    # Get variations
    variations = {
        "Interclass": {layer_name: [] for layer_name in layer_names},
        "Intraclass": {layer_name: [] for layer_name in layer_names},
    }
    for layer_name in layer_names:
        for reseponse in responses:
            label = response["label"]
            activation = response["cache"]["activations"][layer_name][0]
            distance = distance_function(activation, centroids[label][layer_name], args.distance_function)
            for centroid_label in centroids.keys():
                if label == centroid_label:
                    variations["Intraclass"][layer_name].append(distance)
                else:
                    variations["Interclass"][layer_name].append(distance)
        variations["Intraclass"][layer_name] = np.array(variations["Intraclass"][layer_name])
        variations["Interclass"][layer_name] = np.array(variations["Interclass"][layer_name])

    # Plot (x-axis: layer_name, y-axis: variations)
    for class_name, class_variation in variations.items():
        xs = list(range(len(class_variation)))
        ys = [layer_variation for layer_variation in class_variation.values()]
        plt.plot(xs, ys, label=class_name)
        plt.xticks(xs, list(class_variation.keys()), rotation=90)
        plt.xlabel("Layer")
        plt.ylabel("Inter/Intra Variations")
        plt.legend()
    plt.show()
