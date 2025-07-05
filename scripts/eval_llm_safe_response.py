import os
import json
import argparse
from tqdm import tqdm
from iris.data_types import SafeGuardResponse
from iris.model_wrappers.guard_models import load_safeguard, AVAILABLE_GUARDS


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--safeguard_name", type=str, default="LlamaGuard", choices=list(AVAILABLE_GUARDS.keys()))
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-Guard-3-8B")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--api_base", type=str, default=None)
    parser.add_argument("--top_logprobs", type=int, default=2)
    parser.add_argument("--max_tokens", type=int, default=100000)
    parser.add_argument("--disable_logitlens", action="store_true")
    parser.add_argument("--output_path", type=str, default="./outputs/inference_safeguard.jsonl")
    args = parser.parse_args()

    safeguard = load_safeguard(
        safeguard_name=args.safeguard_name,
        model_name=args.model_name,
        checkpoint_path=args.checkpoint_path,
        api_key=args.api_key,
        api_base=args.api_base,
        disable_logitlens=args.disable_logitlens,
        top_logprobs=args.top_logprobs,
        max_tokens=args.max_tokens,
    )

    # Load existing refusal results if available
    count = 0
    if os.path.exists(args.save_path):
        with open(args.save_path, "r") as f:
            for line in f:
                if line.strip() != "":
                    count += 1

    # Load the dataset
    with open(args.load_path, "r") as f:
        for sample_id, line in tqdm(enumerate(f)):
            if sample_id < count:
                continue
            sample = json.loads(line.strip())
            prompt = sample.get("prompt", "")
            responses = sample.get("responses", [])

            safe_responses = []
            for response in responses:
                safeguard_response: SafeGuardResponse = safeguard.predict(prompt=prompt, response=response)
                # print(safeguard_response)
                safe_responses.append(safeguard_response.response_labels)
            sample["safe_response_results"] = safe_responses
            # Save the updated sample
            with open(args.save_path, "a") as out_f:
                out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")