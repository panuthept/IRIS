import os
import torch
import random
import argparse
import numpy as np
from trl import SFTConfig
from peft import LoraConfig
from iris.datasets import WildGuardMixDataset
from iris.model_wrappers.guard_models import WildGuard
from iris.utilities.loaders import load_iris_diff_triplet_config as load_iris_config


"""
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 scripts/iris_diff_triplet_wildguard.py \
--iris_config ./data/iris_diff_triplet_configs/layer_1_to_10.json \
--iris_label_path ./data/saved_directions \
--model_name allenai/wildguard \
--train_eval_split 0.9 \
--max_seq_length 2048 \
--batch_size 1 \
--gradient_accumulation_steps 32 \
--epochs 2 \
--eval_steps 60 \
--save_total_limit 100 \
--output_dir ./finetuned_models/iris_diff_triplet_wildguard_layer_1_to_10 \
--use_lora \
--lora_rank 128

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 scripts/iris_diff_triplet_wildguard.py \
--iris_config ./data/iris_diff_triplet_configs/layer_11_to_18.json \
--iris_label_path ./data/saved_directions \
--model_name allenai/wildguard \
--train_eval_split 0.9 \
--max_seq_length 2048 \
--batch_size 1 \
--gradient_accumulation_steps 32 \
--epochs 2 \
--eval_steps 60 \
--save_total_limit 100 \
--output_dir ./finetuned_models/iris_diff_triplet_wildguard_layer_11_to_18 \
--use_lora \
--lora_rank 128
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iris_config", type=str, default="./data/iris_diff_triplet_configs/layer_1_to_10.json")
    parser.add_argument("--iris_label_path", type=str, default="./data/saved_directions")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--attack_engine", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default="./data/datasets/wildguardmix")
    parser.add_argument("--train_eval_split", type=float, default=0.9)
    parser.add_argument("--max_seq_length", type=int, default=8192)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--label_smoothing_factor", type=float, default=0.00)
    parser.add_argument("--learning_rate", type=float, default=2e-6)
    parser.add_argument("--weight_decay", type=float, default=0.00)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--eval_steps", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="./finetuned_models/iris_diff_triplet_wildguard")
    parser.add_argument("--report_to", type=str, default="all")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--allow_cpu", action="store_true")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--save_total_limit", type=int, default=1)
    args = parser.parse_args()

    random.seed(args.seed)

    model = WildGuard(
        model_name_or_path=args.model_name, 
        disable_logitlens=False,
        enable_logitlens_cache=False,
    )
    dataset = WildGuardMixDataset(attack_engine=args.attack_engine, cache_dir=args.cache_dir)
    samples = dataset.as_samples(split="train")

    # Load config for the intermediate_labels
    label_mapper = {"Harmful": 5849, "Benign": 1476}
    iris_config = load_iris_config(args.iris_config)
    iris_labels = {}
    for path in os.listdir(args.iris_label_path):
        module_name, final_label, bin = path.split("_")
        bin = bin.replace(".npy", "")
        if bin != "3":
            continue
        with open(os.path.join(args.iris_label_path, path), "rb") as f:
            data = np.load(f)   # shape: (1, embedding_dim)
            if module_name not in iris_labels:
                iris_labels[module_name] = {}
            iris_labels[module_name][label_mapper[final_label]] = data
    iris_config.layer_labels = iris_labels
    print(iris_config)

    random.shuffle(samples)
    if args.train_eval_split == 1.0:
        train_samples, eval_samples = samples, []
    else:
        train_size = int(len(samples) * args.train_eval_split)
        train_samples, eval_samples = samples[:train_size], samples[train_size:]
    # Log the number of samples in the train and eval datasets
    print(f"Train size: {len(train_samples)}")
    print(f"Eval size: {len(eval_samples)}")

    is_gpu_available = torch.cuda.is_available()
    if is_gpu_available or args.allow_cpu:
        print(f"GPU Count: {torch.cuda.device_count()}")
        do_eval = len(eval_samples) > 0
        model.train_iris_diff_triplet(
            train_samples=train_samples,
            eval_samples=eval_samples,
            iris_config=iris_config,
            sft_config=SFTConfig(
                output_dir=args.output_dir, 
                report_to=args.report_to,
                max_seq_length=args.max_seq_length,
                per_device_train_batch_size=args.batch_size,
                per_device_eval_batch_size=args.batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                label_smoothing_factor=args.label_smoothing_factor,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                warmup_ratio=args.warmup_ratio,
                num_train_epochs=args.epochs,
                eval_strategy="steps",
                logging_strategy="steps",
                logging_steps=10,
                eval_steps=args.eval_steps,
                save_steps=args.eval_steps,
                save_total_limit=args.save_total_limit,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                overwrite_output_dir=True,
                do_train=True,
                do_eval=do_eval,
                do_predict=False,
                seed=args.seed,
                bf16=args.bf16,
                fp16=args.fp16,
            ),
            peft_config=LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            ) if args.use_lora else None,
        )