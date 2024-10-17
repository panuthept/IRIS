import json
import argparse
from tqdm import tqdm
from iris.datasets import load_dataset
from iris.model_wrappers.guard_models import WildGuard


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--checkpoint_path", type=str, default="./finetuned_models/sft_wildguard/checkpoint-1220")
    parser.add_argument("--dataset_name", type=str, default="JailbreakBenchDataset")
    parser.add_argument("--prompt_intention", type=str, default=None)
    parser.add_argument("--attack_engine", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="test")
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

    with open(args.output_path, "w") as f:
        for sample in tqdm(samples):
            prompts = sample.get_prompts()
            labels = sample.instructions_true_label
            for prompt, label in zip(prompts, labels):
                response = model.generate(prompt, return_probs=True)
                cache = model.model.logitlens.fetch_cache(return_tokens=False, return_logits=True, return_activations=True)
                f.write(json.dumps({
                    "prompt": prompt,
                    "response": response,
                    "label": label,
                    "cache": cache
                }, ensure_ascii=False) + "\n")