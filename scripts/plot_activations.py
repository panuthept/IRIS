import json
import argparse
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 

from tqdm import tqdm
from transformers import AutoTokenizer


def load_examples(
    load_path, 
    k: int = 5,
    layer_nums: int = 32,
    max_examples: int = None,
    prompt_intention: str = "Harmful",
    correct_prediction_only: bool = False, 
    incorrect_prediction_only: bool = False,
    activation_template: str = "model.layers.{layer_id}",
    ):
    assert (correct_prediction_only and incorrect_prediction_only) == False, \
        "Only one of correct_prediction_only and incorrect_prediction_only can be True"
    
    modules = set([activation_template.format(layer_id=i) for i in range(layer_nums)] + ["final_predictions"])

    examples = []
    prompts = set()
    with open(load_path, "r") as f:
        for line in tqdm(f):
            example = json.loads(line)
            if example["label"] != prompt_intention:
                continue
            if correct_prediction_only and example["response"] != example["label"]:
                continue
            if incorrect_prediction_only and example["response"] == example["label"]:
                continue
            if example["prompt"] in prompts:
                continue
            prompts.add(example["prompt"])

            activations = {}
            for module, activation in example["cache"]["logits"].items():
                if module in modules:
                    activations[module] = activation[0][:k]
            example = {"prompt": example["prompt"], "label": example["label"], "response": example["response"], "activations": activations}

            examples.append(example)
            if max_examples is not None and len(examples) >= max_examples:
                break
    return examples

def load_tokenizer(model_name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir="./data/models",
            local_files_only=True,
        )
    except:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir="./data/models",
            local_files_only=False,
        )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path_1", type=str, required=True)
    parser.add_argument("--load_path_2", type=str, default=None)
    parser.add_argument("--prompt_intention", type=str, default="Harmful")
    parser.add_argument("--model_name", type=str, default="allenai/wildguard")
    parser.add_argument("--correct_prediction_only", action="store_true")
    parser.add_argument("--incorrect_prediction_only", action="store_true")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--layer_nums", type=int, default=32)
    parser.add_argument("--activation_template", type=str, default="model.layers.{layer_id}")
    args = parser.parse_args()

    examples = load_examples(
        args.load_path_1,
        k=args.k,
        layer_nums=args.layer_nums,
        max_examples=args.max_examples,
        prompt_intention=args.prompt_intention,
        correct_prediction_only=args.correct_prediction_only,
        incorrect_prediction_only=args.incorrect_prediction_only,
        activation_template=args.activation_template,
    )
    if args.load_path_2:
        examples += load_examples(
            args.load_path_2,
            k=args.k,
            layer_nums=args.layer_nums,
            max_examples=args.max_examples,
            prompt_intention=args.prompt_intention,
            correct_prediction_only=args.correct_prediction_only,
            incorrect_prediction_only=args.incorrect_prediction_only,
            activation_template=args.activation_template,
        )

    tokenizer = load_tokenizer(args.model_name)

    count = 0
    accum_activations = {}
    for example in examples:
        # print(example["prompt"])
        # print("=" * 100)
        activations = example["activations"]
        # if activations["model.layers.17"][0] == "▁yes":
        #     continue
        # print(example["prompt"])
        # print("-" * 100)
        for module_name, module_activations in activations.items():
            # if module_name == "model.layers.17":
            #     if (
            #         module_activations[0][0] == "▁yes" and \
            #         module_activations[1][0] == "yes" and \
            #         module_activations[2][0] == "▁Yes" and \
            #         module_activations[3][0] == "Yes" and \
            #         module_activations[4][0] == "▁outrage"
            #     ):
            #         print(example["prompt"])
            #         print(module_activations)
            #         print("-" * 100)
            if module_name not in accum_activations:
                accum_activations[module_name] = {}
            for rank, (token, logit) in enumerate(module_activations):
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
            token_str = tokenizer._convert_id_to_token(token)
            freq_matrix[rank][layer_idx] = freq
            labels.append(f"{token_str}\n({token})\n{freq}")
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
    plt.close()