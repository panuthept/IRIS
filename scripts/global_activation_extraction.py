import os
import json
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.decomposition import PCA


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", type=str, default="./data/activations/activations_and_logits.jsonl")
    parser.add_argument("--save_path", type=str, default="./data/activations")
    parser.add_argument("--token_depth", type=int, default=1)
    parser.add_argument("--plot_layer_name", type=str, default="model.layers.19")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    # Get majority tokens for each module
    if not os.path.exists(os.path.join(args.save_path, f"majority_tokens_{args.token_depth}.pkl")):
        token_freqs = {}
        with open(args.load_path, "r") as f:
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
        # Save majority tokens
        with open(os.path.join(args.save_path, f"majority_tokens_{args.token_depth}.pkl"), "wb") as f:
            pickle.dump(majority_tokens, f)
    else:
        with open(os.path.join(args.save_path, f"majority_tokens_{args.token_depth}.pkl"), "rb") as f:
            majority_tokens = pickle.load(f)

    # Get global activations
    if not os.path.exists(os.path.join(args.save_path, f"global_activations_{args.token_depth}.pkl")):
        class_activations = {}
        global_activations = {}
        with open(args.load_path, "r") as f:
            for line in tqdm(f):
                example = json.loads(line)
                label = example["label"]
                response = example["response"][0][0]
                if response != label:
                    continue
                for module_name in example["cache"]["activations"].keys():
                    # Get sequence of tokens
                    tokens = tuple([token_id for token_id, score in example["cache"]["logits"][module_name][0][:args.token_depth]])
                    # Collect class activations
                    if args.plot:
                        if module_name not in class_activations:
                            class_activations[module_name] = {}
                        if label not in class_activations[module_name]:
                            class_activations[module_name][label] = []
                        class_activations[module_name][label].append(example["cache"]["activations"][module_name][0])
                    # Update global activations
                    if tokens != majority_tokens[module_name][label]:
                        continue
                    if module_name not in global_activations:
                        global_activations[module_name] = {}
                    if label not in global_activations[module_name]:
                        global_activations[module_name][label] = []
                    global_activations[module_name][label].append(example["cache"]["activations"][module_name][0])
        # Average activations
        global_activations = {module_name: {label: np.mean(activations, axis=0, keepdims=True) for label, activations in label_activations.items()} for module_name, label_activations in global_activations.items()}
        # Save global activations
        with open(os.path.join(args.save_path, f"global_activations_{args.token_depth}.pkl"), "wb") as f:
            pickle.dump(global_activations, f)
        with open(os.path.join(args.save_path, "class_activations.pkl"), "wb") as f:
            pickle.dump(class_activations, f)
    else:
        with open(os.path.join(args.save_path, f"global_activations_{args.token_depth}.pkl"), "rb") as f:
            global_activations = pickle.load(f)
        with open(os.path.join(args.save_path, "class_activations.pkl"), "rb") as f:
            class_activations = pickle.load(f)

    # Plot activations
    if args.plot:
        for layer_num in range(32):
            layer_name = f"model.layers.{layer_num}"
            print(majority_tokens[layer_name])
            # Convert global_activations and class_activations to Numpy array
            plot_global_activations = {module_name: {label: np.array(activations) for label, activations in label_activations.items()} for module_name, label_activations in global_activations.items()}[layer_name]
            plot_class_activations = {module_name: {label: np.array(activations) for label, activations in label_activations.items()} for module_name, label_activations in class_activations.items()}[layer_name]
            # Reduce activations dimension to 2D
            pca = PCA(n_components=2)
            pca.fit(np.concatenate([activations for activations in plot_class_activations.values()], axis=0))
            plot_global_activations = {label: pca.transform(activations) for label, activations in plot_global_activations.items()}
            plot_class_activations = {label: pca.transform(activations) for label, activations in plot_class_activations.items()}
            # Plot activations
            plt.subplot(4, 8, layer_num + 1)
            for label, activations in plot_class_activations.items():
                plt.scatter(activations[:, 0], activations[:, 1], label=label, alpha=0.2, marker="o", color="orange" if label == "Harmful" else "green")
            for label, activations in plot_global_activations.items():
                plt.scatter(activations[:, 0], activations[:, 1], label=f"{label} (Global)", s=30, alpha=1.0, marker="^", color="red" if label == "Harmful" else "blue")
            for label, activations in plot_class_activations.items():
                plt.scatter(activations[:, 0].mean(axis=0, keepdims=True), activations[:, 1].mean(axis=0, keepdims=True), label=f"{label} (Mean)", s=30, alpha=1.0, marker="s", color="red" if label == "Harmful" else "blue")
            # plt.legend()
            plt.title(layer_name, fontsize=9)
            # No axis 
            plt.xticks([])
            plt.yticks([])
        plt.tight_layout()
        plt.show()