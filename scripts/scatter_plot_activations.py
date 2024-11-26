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
    parser.add_argument("--n_components", type=int, default=2)
    parser.add_argument("--plot_layer_name", type=str, default=None)
    args = parser.parse_args()

    class_activations = {}
    with open(args.load_path, "r") as f:
        for line in tqdm(f):
            example = json.loads(line)
            label = example["label"]
            response = example["response"][0][0]
            category = f"{label} -> {response}"
            for module, activation in example["cache"]["activations"].items():
                if module not in class_activations:
                    class_activations[module] = {}
                if category not in class_activations[module]:
                    class_activations[module][category] = []
                class_activations[module][category].append(activation[0])

    colors = {
        "Harmful -> Harmful": "orange",
        "Harmful -> Benign": "red",
        "Benign -> Benign": "green",
        "Benign -> Harmful": "blue",
    }

    # Plot activations
    if args.plot_layer_name is None:
        for layer_num in range(32):
            layer_name = f"model.layers.{layer_num}"
            # Convert global_activations and class_activations to Numpy array
            plot_class_activations = {module_name: {label: np.array(activations) for label, activations in label_activations.items()} for module_name, label_activations in class_activations.items()}[layer_name]
            # Reduce activations dimension to 2D
            pca = PCA(n_components=2)
            pca.fit(np.concatenate([activations for activations in plot_class_activations.values()], axis=0))
            plot_class_activations = {label: pca.transform(activations) for label, activations in plot_class_activations.items()}
            # Plot activations
            plt.subplot(4, 8, layer_num + 1)
            for label, activations in plot_class_activations.items():
                plt.scatter(activations[:, 0], activations[:, 1], label=label, alpha=0.2, marker="o", color=colors[label])
            # plt.legend()
            plt.title(layer_name, fontsize=9)
            # No axis 
            plt.xticks([])
            plt.yticks([])
        plt.tight_layout()
        plt.show()
    else:
        layer_name = args.plot_layer_name
        # Convert global_activations and class_activations to Numpy array
        plot_class_activations = {module_name: {label: np.array(activations) for label, activations in label_activations.items()} for module_name, label_activations in class_activations.items()}[layer_name]
        # Reduce activations dimension to 2D
        pca = PCA(n_components=args.n_components)
        pca.fit(np.concatenate([activations for activations in plot_class_activations.values()], axis=0))
        plot_class_activations = {label: pca.transform(activations) for label, activations in plot_class_activations.items()}
        # Plot activations
        fig = plt.figure()
        if args.n_components == 2:
            ax = fig.add_subplot()
        elif args.n_components == 3:
            ax = fig.add_subplot(projection='3d')
        for label, activations in plot_class_activations.items():
            if args.n_components == 2:
                ax.scatter(activations[:, 0], activations[:, 1], label=label, alpha=0.2, marker="o", color=colors[label])
            elif args.n_components == 3:
                ax.scatter(activations[:, 0], activations[:, 1], activations[:, 2], label=label, alpha=0.2, marker="o", color=colors[label])
        # plt.legend()
        plt.title(layer_name)
        # No axis 
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.show()