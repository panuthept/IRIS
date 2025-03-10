import os
import random
import argparse
import numpy as np
from tqdm import tqdm
from iris.datasets import load_dataset, AVAILABLE_DATASETS
from iris.model_wrappers.guard_models import load_safeguard, AVAILABLE_GUARDS

"""
CUDA_VISIBLE_DEVICES=0 python scripts/tokenize_safeguard.py \
--safeguard_name WildGuard \
--model_name allenai/wildguard \
--dataset_name SEASafeguardDataset \
--dataset_split test \
--language en
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--safeguard_name", type=str, default="ShieldGemma", choices=list(AVAILABLE_GUARDS.keys()))
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-Guard-3-1B")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="WildGuardMixDataset", choices=list(AVAILABLE_DATASETS.keys()))
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top_logprobs", type=int, default=128)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--prompt_intention", type=str, default=None)
    parser.add_argument("--attack_engine", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="test")
    args = parser.parse_args()

    random.seed(args.seed)

    # Initial safeguard
    safeguard = load_safeguard(
        safeguard_name=args.safeguard_name,
        model_name=args.model_name,
        checkpoint_path=args.checkpoint_path,
        disable_logitlens=True,
        top_logprobs=args.top_logprobs,
    )

    # Initial dataset
    dataset = load_dataset(args.dataset_name, args.prompt_intention, args.attack_engine, args.language)
    samples = dataset.as_samples(split=args.dataset_split)
    random.shuffle(samples)

    if args.max_samples is not None:
        samples = samples[:args.max_samples]

    # Create save directory
    # os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    seq_lengths = []
    for sample in tqdm(samples):
        prompts = sample.get_prompts()
        _gold_labels = sample.instructions_true_label
        for prompt, gold_label in zip(prompts, _gold_labels):
            encoded_inputs = safeguard.tokenize(
                prompt=prompt, 
            )
            seq_lengths.append(encoded_inputs["input_ids"].shape[-1])
    seq_lengths = np.array(seq_lengths)
    print(f"Max sequence length: {seq_lengths.max()}")
    print(f"Min sequence length: {seq_lengths.min()}")
    print(f"Mean sequence length: {seq_lengths.mean()}")
    print(f"Median sequence length: {np.median(seq_lengths)}")
    print(f"Standard deviation of sequence length: {seq_lengths.std()}")
    print(f"99th percentile of sequence length: {np.percentile(seq_lengths, 99)}")