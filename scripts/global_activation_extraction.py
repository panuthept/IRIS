import json
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_activation_path", type=str, default="./data/activations/activations_and_logits.jsonl")
    parser.add_argument("--save_activation_path", type=str, default="./data/activations/global_activations.json")
    parser.add_argument("--token_depth", type=int, default=1)
    parser.add_argument("--plot_layer_name", type=str, default="model.layers.19")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    # Get majority tokens for each module
    token_freqs = {}
    with open(args.load_activation_path, "r") as f:
        for line in tqdm(f):
            example = json.loads(line)
            label = example["label"]
            response = example["response"][0][0]
            if response != label:
                continue
            for module_name, logits in example["cache"]["logits"].items():
                # Get sequence of tokens
                tokens = tuple([token_id for token_id, score in logits[0][:args.token_depth]])
                # Initialize token frequency dictionary
                if module_name not in token_freqs:
                    token_freqs[module_name] = {}
                if label not in token_freqs[module_name]:
                    token_freqs[module_name][label] = {}
                # Update token frequency dictionary
                token_freqs[module_name][label][tokens] = token_freqs[module_name][label].get(tokens, 0) + 1
    # Normalize the frequency of tokens
    token_freqs = {module_name: {label: {tokens: freq / sum(token_freqs[module_name][label].values()) for tokens, freq in tokens.items()} for label, tokens in label_tokens.items()} for module_name, label_tokens in token_freqs.items()}
    # Get majority tokens
    majority_tokens = {module_name: {label: sorted(tokens.items(), key=lambda x: x[1], reverse=True)[0][0] for label, tokens in label_tokens.items()} for module_name, label_tokens in token_freqs.items()}
    print(majority_tokens)

    # Get global activations
    class_activations = {}
    global_activations = {}
    with open(args.load_activation_path, "r") as f:
        for line in tqdm(f):
            example = json.loads(line)
            label = example["label"]
            response = example["response"][0][0]
            if response != label:
                continue
            for module_name in example["cache"]["activations"].keys():
                # Get sequence of tokens
                tokens = tuple([token_id for token_id, score in example["cache"]["logits"][module_name][0][:args.token_depth]])
                if tokens != majority_tokens[module_name][label]:
                    continue
                # Update global activations
                if module_name not in global_activations:
                    global_activations[module_name] = {}
                if label not in global_activations[module_name]:
                    global_activations[module_name][label] = []
                global_activations[module_name][label].append(example["cache"]["activations"][module_name][0])
                # Collect class activations
                if args.plot and module_name == args.plot_layer_name:
                    if label not in class_activations:
                        class_activations[label] = []
                    class_activations[label].append(example["cache"]["activations"][module_name][0])
    # Average activations
    global_activations = {module_name: {label: np.mean(activations, axis=0, keepdims=True) for label, activations in label_activations.items()} for module_name, label_activations in global_activations.items()}
    # print(global_activations)

    # Plot activations
    if args.plot:
        # Convert global_activations and class_activations to Numpy array
        plot_global_activations = {module_name: {label: np.array(activations) for label, activations in label_activations.items()} for module_name, label_activations in global_activations.items()}[args.plot_layer_name]
        plot_class_activations = {label: np.array(activations) for label, activations in class_activations.items()}
        # Reduce activations dimension to 2D
        pca = PCA(n_components=2)
        pca.fit(np.concatenate([activations for activations in plot_class_activations.values()], axis=0))
        plot_global_activations = {label: pca.transform(activations) for label, activations in plot_global_activations.items()}
        plot_class_activations = {label: pca.transform(activations) for label, activations in plot_class_activations.items()}
        # Plot activations
        plt.figure()
        for label, activations in plot_class_activations.items():
            plt.scatter(activations[:, 0], activations[:, 1], label=label, alpha=0.3, marker="o", color="orange" if label == "Harmful" else "green")
        for label, activations in plot_global_activations.items():
            plt.scatter(activations[:, 0], activations[:, 1], label=f"{label} (Global)", alpha=1.0, marker="^", color="red" if label == "Harmful" else "blue")
        plt.legend()
        plt.show()