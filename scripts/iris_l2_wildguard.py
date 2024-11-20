import os
import torch
import random
import pickle
import argparse
from trl import SFTConfig
from peft import LoraConfig
from iris.datasets import WildGuardMixDataset
from iris.utilities.loaders import load_iris_l2_config
from iris.model_wrappers.guard_models import WildGuard


####################### SFT with LoRA #######################
# CUDA_VISIBLE_DEVICES=0,1 accelerate launch scripts/sft_wildguard.py --model_name mistralai/Mistral-7B-v0.3 --train_eval_split 0.9 --max_seq_length 2048 --batch_size 1 --gradient_accumulation_steps 64 --epochs 2 --eval_steps 60 --output_dir ./finetuned_models/sft_wildguard --use_lora
# accelerate launch scripts/sft_wildguard.py --model_name mistralai/Mistral-7B-v0.3 --train_eval_split 0.9 --max_seq_length 2048 --batch_size 1 --gradient_accumulation_steps 64 --epochs 2 --eval_steps 60 --output_dir ./finetuned_models/sft_wildguard_vanilla --use_lora --attack_engine vanilla
# accelerate launch scripts/sft_wildguard.py --model_name mistralai/Mistral-7B-v0.3 --train_eval_split 0.9 --max_seq_length 2048 --batch_size 1 --gradient_accumulation_steps 64 --epochs 2 --eval_steps 60 --output_dir ./finetuned_models/sft_wildguard_adversarial --use_lora --attack_engine adversarial
##################### SFT Full-finetune #####################
# CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch scripts/sft_wildguard.py --model_name mistralai/Mistral-7B-v0.3 --train_eval_split 0.9 --max_seq_length 4096 --batch_size 1 --gradient_accumulation_steps 42 --epochs 2 --eval_steps 60 --output_dir ./finetuned_models/sft_wildguard
# accelerate launch scripts/sft_wildguard.py --model_name mistralai/Mistral-7B-v0.3 --train_eval_split 0.9 --max_seq_length 2048 --batch_size 1 --gradient_accumulation_steps 64 --epochs 2 --eval_steps 60 --output_dir ./finetuned_models/sft_wildguard_vanilla --attack_engine vanilla
# accelerate launch scripts/sft_wildguard.py --model_name mistralai/Mistral-7B-v0.3 --train_eval_split 0.9 --max_seq_length 2048 --batch_size 1 --gradient_accumulation_steps 64 --epochs 2 --eval_steps 60 --output_dir ./finetuned_models/sft_wildguard_adversarial --attack_engine adversarial


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iris_config", type=str, default="./data/iris_l2_configs/benign_only_configs/layer_19.json")
    parser.add_argument("--global_activation_path", type=str, default="./data/activations/global_activations_1.pkl")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--attack_engine", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default="./data/datasets/wildguardmix")
    parser.add_argument("--train_eval_split", type=float, default=0.9)
    parser.add_argument("--max_seq_length", type=int, default=8192)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-6)
    parser.add_argument("--weight_decay", type=float, default=0.00)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--eval_steps", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="./finetuned_models/sft_wildguard")
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
    iris_config = load_iris_l2_config(args.iris_config)
    with open(args.global_activation_path, "rb") as f:
        global_activations = pickle.load(f)
        global_activations = {module_name: {label_mapper[label]: activations for label, activations in label_activations.items()} for module_name, label_activations in global_activations.items()}
    iris_config.layer_labels = global_activations
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
        model.train_iris_l2(
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