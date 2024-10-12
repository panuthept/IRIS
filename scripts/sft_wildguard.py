import torch
import argparse
from trl import SFTConfig
from iris.datasets import WildGuardMixDataset
from iris.model_wrappers.guard_models import WildGuard


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-v0.3")
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
    parser.add_argument("--allow_cpu", action="store_true")
    args = parser.parse_args()

    model = WildGuard(model_name_or_path=args.model_name)
    dataset = WildGuardMixDataset(cache_dir=args.cache_dir)

    samples = dataset.as_samples(split="train")
    train_size = int(len(samples) * args.train_eval_split)
    train_samples, eval_samples = samples[:train_size], samples[train_size:]
    # Log the number of samples in the train and eval datasets
    print(f"Train size: {len(train_samples)}")
    print(f"Eval size: {len(eval_samples)}")

    is_gpu_available = torch.cuda.is_available()
    if is_gpu_available or args.allow_cpu:
        print(f"GPU Count: {torch.cuda.device_count()}")
        do_eval = len(eval_samples) > 0
        model.train_sft(
            train_samples=train_samples,
            eval_samples=eval_samples,
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
                save_total_limit=12,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                overwrite_output_dir=True,
                do_train=True,
                do_eval=do_eval,
                do_predict=False,
                seed=args.seed,
            ),
        )