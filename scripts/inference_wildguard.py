import os
import json
import argparse
from tqdm import tqdm
from iris.datasets import load_dataset
from iris.model_wrappers.guard_models import WildGuard

"""
CUDA_VISIBLE_DEVICES=0 python scripts/inference_wildguard.py \
--model_name allenai/wildguard \
--checkpoint_path ./finetuned_models/iris_l2_wildguard_layer_19_b05h05_10_epochs_v2/checkpoint-6100 \
--dataset_name ORBenchDataset \
--dataset_split test \
--prompt_intention hard_benign \
--output_path ./outputs/iris_l2_wildguard_layer_19_b05h05_10_epochs_v2/checkpoint-6100/ORBenchDataset/test/hard_benign_prompts.jsonl

CUDA_VISIBLE_DEVICES=1 python scripts/inference_wildguard.py \
--model_name allenai/wildguard \
--checkpoint_path ./finetuned_models/iris_l2_wildguard_layer_19_b05h05_10_epochs_v2/checkpoint-6100 \
--dataset_name ORBenchDataset \
--dataset_split test \
--prompt_intention harmful \
--output_path ./outputs/iris_l2_wildguard_layer_19_b05h05_10_epochs_v2/checkpoint-6100/ORBenchDataset/test/harmful_prompts.jsonl

CUDA_VISIBLE_DEVICES=1 python scripts/inference_wildguard.py \
--model_name allenai/wildguard \
--checkpoint_path ./finetuned_models/iris_l2_wildguard_layer_19_benign_only_10_epochs_v2/checkpoint-2520 \
--dataset_name ORBenchDataset \
--dataset_split test \
--prompt_intention harmful \
--output_path ./outputs/iris_l2_wildguard_layer_19_benign_only_10_epochs_v2/checkpoint-2520/ORBenchDataset/test/harmful_prompts.jsonl

CUDA_VISIBLE_DEVICES=2 python scripts/inference_wildguard.py \
--model_name allenai/wildguard \
--checkpoint_path ./finetuned_models/iris_l2_wildguard_layer_19_benign_only_10_epochs_v2/checkpoint-3780 \
--dataset_name ORBenchDataset \
--dataset_split test \
--prompt_intention harmful \
--output_path ./outputs/iris_l2_wildguard_layer_19_benign_only_10_epochs_v2/checkpoint-3780/ORBenchDataset/test/harmful_prompts.jsonl

CUDA_VISIBLE_DEVICES=3 python scripts/inference_wildguard.py \
--model_name allenai/wildguard \
--checkpoint_path ./finetuned_models/iris_l2_wildguard_layer_19_benign_only_10_epochs_v2/checkpoint-5040 \
--dataset_name ORBenchDataset \
--dataset_split test \
--prompt_intention harmful \
--output_path ./outputs/iris_l2_wildguard_layer_19_benign_only_10_epochs_v2/checkpoint-5040/ORBenchDataset/test/harmful_prompts.jsonl
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="allenai/wildguard")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="WildGuardMixDataset")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--prompt_intention", type=str, default=None)
    parser.add_argument("--attack_engine", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--save_tokens", action="store_true")
    parser.add_argument("--save_logits", action="store_true")
    parser.add_argument("--save_activations", action="store_true")
    parser.add_argument("--output_path", type=str, default="./outputs/inference_wildguard.jsonl")
    args = parser.parse_args()

    # Initial model
    model = WildGuard(
        model_name_or_path=args.model_name,
        checkpoint_path=args.checkpoint_path,
    )

    # Initial dataset
    dataset = load_dataset(args.dataset_name, args.prompt_intention, args.attack_engine)
    samples = dataset.as_samples(split=args.dataset_split)
    if args.max_samples is not None:
        samples = samples[:args.max_samples]

    # Create save directory
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    tp = 0
    fp = 0
    harmful_count = 0
    benign_count = 0
    with open(args.output_path, "w") as f:
        for sample in tqdm(samples):
            prompts = sample.get_prompts()
            gold_labels = sample.instructions_true_label
            for prompt, gold_label in zip(prompts, gold_labels):
                response = model.generate(prompt, return_probs=True)

                classified_label = sorted(response, key=lambda x: x[1], reverse=True)[0][0]
                if gold_label == "Harmful":
                    harmful_count += 1
                    if classified_label == "Harmful":
                        tp += 1
                else:
                    benign_count += 1
                    if classified_label == "Harmful":
                        fp += 1

                cache = model.model.logitlens.fetch_cache(return_tokens=args.save_tokens, return_logits=args.save_logits, return_activations=args.save_activations)
                cache = {key: {module_name: activation for module_name, activation in activations.items() if "self_attn" not in module_name and "mlp" not in module_name} for key, activations in cache.items()}
                f.write(json.dumps({
                    "prompt": prompt,
                    "response": response,
                    "label": gold_label,
                    "cache": cache
                }, ensure_ascii=False) + "\n")
    # Calculate TPR and FPR
    tpr = tp / (harmful_count + 1e-7)
    fpr = fp / (benign_count + 1e-7)
    print(f"TPR: {round(tpr, 2)}")
    print(f"FPR: {round(fpr, 2)}")