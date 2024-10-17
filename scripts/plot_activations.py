import json
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 


if __name__ == "__main__":
    examples = []
    # with open("./benign_prompts.jsonl", "r") as f:
    #     for line in f:
    #         example = json.loads(line)
    #         if example["response"] == example["label"]:
    #             examples.append(example)
    #         # examples.append(example)
    # examples = examples[:500]
    # benign_count = len(examples)
    # print(f"Benigns: {benign_count}")

    with open("./harmful_prompts.jsonl", "r") as f:
        for line in f:
            example = json.loads(line)
            if example["response"] == example["label"]:
                examples.append(example)
            # examples.append(example)
    examples = examples[:500]

    # with open("./adv_benign_prompts.jsonl", "r") as f:
    #     for line in f:
    #         example = json.loads(line)
    #         if example["response"] != example["label"]:
    #             examples.append(example)
    #         # examples.append(example)

    # with open("./adv_harmful_prompts.jsonl", "r") as f:
    #     for line in f:
    #         example = json.loads(line)
    #         # if example["response"] == example["label"]:
    #         #     examples.append(example)
    #         examples.append(example)
 
    modules = [f"model.layers.{i}" for i in range(32)] + ["final_predictions"]
    # modules = [f"model.layers.{i}.mlp" for i in range(32)] + ["final_predictions"]
    # modules = [f"model.layers.{i}.self_attn" for i in range(32)] + ["final_predictions"]

    count = 0
    accum_activations = {}
    for example in examples:
        activations = example["activations"]
        # if activations["model.layers.17"][0] == "▁yes":
        #     continue
        # print(example["prompt"])
        # print("-" * 100)
        for module_name, module_activations in activations.items():
            if module_name not in modules:
                continue
            if module_name not in accum_activations:
                accum_activations[module_name] = {}
            for rank, token in enumerate(module_activations):
                if rank not in accum_activations[module_name]:
                    accum_activations[module_name][rank] = {}
                if token not in accum_activations[module_name][rank]:
                    accum_activations[module_name][rank][token] = 0
                accum_activations[module_name][rank][token] += 1
        count += 1
    print(f"Number of Examples: {count}")
    # print(accum_activations)

    sorted_activations = {}
    for module_name in accum_activations:
        selected_tokens = set()
        sorted_activations[module_name] = []
        for rank in accum_activations[module_name]:
            for selected_token, freq in sorted(accum_activations[module_name][rank].items(), key=lambda x: x[1], reverse=True):
                if selected_token not in selected_tokens:
                    sorted_activations[module_name].append((selected_token, round(freq / count, 2)))
                    selected_tokens.add(selected_token)
                    break

    selected_activations = {module_name: activations for module_name, activations in sorted_activations.items()}
    # print(selected_activations)

    freq_matrix = np.zeros([5, len(selected_activations)])
    label_matrix = []
    for layer_idx, (module_name, activations) in enumerate(selected_activations.items()):
        labels = []
        for rank, (token, freq) in enumerate(activations):
            freq_matrix[rank][layer_idx] = freq
            labels.append(f"{token}\n{freq}")
        label_matrix.append(labels)
    label_matrix = np.asarray(label_matrix).T

    split_layer = 16

    plt.figure(figsize=(16, 8))
    plt.subplot(2, 1, 1)
    hm = sns.heatmap(
        data=freq_matrix[:, :split_layer], 
        annot=label_matrix[:, :split_layer], 
        fmt="", 
        annot_kws={'size': 10},
        yticklabels=list(range(1, 6)),
        xticklabels=list(range(1, 33))[:split_layer],
        linewidth=.2,
    ) 
    plt.subplot(2, 1, 2)
    hm = sns.heatmap(
        data=freq_matrix[:, split_layer:], 
        annot=label_matrix[:, split_layer:], 
        fmt="", 
        annot_kws={'size': 10},
        yticklabels=list(range(1, 6)),
        xticklabels=list(range(1, 33))[split_layer:] + ["Final"],
        linewidth=.2,
    ) 
    plt.tight_layout()
    plt.show()