import os
import json
import torch
import random
import hashlib
import argparse
from tqdm import tqdm
# from google import genai
from openai import OpenAI
from typing import List, Dict
from iris.data_types import SafeGuardInput
from iris.datasets import load_dataset, AVAILABLE_DATASETS
from transformers import AutoModelForCausalLM, AutoTokenizer


def hash_prompt(prompt: str) -> str:
    """Generate a sha256 hash for the prompt."""
    return hashlib.sha256(prompt.encode('utf-8')).hexdigest()


class APIModel:
    def __init__(
            self, 
            model_name: str, 
            api_key: str = None, 
            api_base: str = None,
    ):
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, base_url=api_base)

    def complete(self, prompt: str = None, messages: List[Dict] = None):
        if messages is not None:
            outputs = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=2048,
            )
            response = outputs.choices[0].message.content
        else:
            outputs = self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=2048,
            )
            response = outputs.choices[0].text
        return response
    
class GeminiAPIModel:
    def __init__(self, model_name: str, project: str, location: str):
        self.model_name = model_name
        self.client = genai.Client(
            vertexai=True, project=project, location=location,
        )

    def complete(self, prompt: str = None, messages: List[Dict] = None):
        assert messages is None, "Gemini API does not support messages format. Please use prompt format instead."
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[prompt]
        )
        return response.text
    
class HFModel:
    def __init__(self, model_name: str, tokenizer_name: str = None):
        self.model_name = model_name
        self.tokenizer = self.load_tokenizer(model_name) if tokenizer_name is None else self.load_tokenizer(tokenizer_name)
        self.model = self.load_model(model_name)

    def load_tokenizer(self, model_name_or_path: str):
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                cache_dir="./data/models",
                local_files_only=True,
            )
        except:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                cache_dir="./data/models",
                local_files_only=False,
            )
        return tokenizer
    
    def load_model(self, model_name_or_path: str):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                cache_dir="./data/models",
                tp_plan="auto",
                local_files_only=True,
            )
        except:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    cache_dir="./data/models",
                    tp_plan="auto",
                    local_files_only=False,
                )
            except:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    cache_dir="./data/models",
                    device_map="auto",
                    local_files_only=False,
                )
        model.eval()  # Set the model to evaluation mode
        return model

    def complete(self, prompt: str = None, messages: List[Dict] = None):
        if messages is not None:
            prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True, 
                return_dict=False
            )
        inputs = self.tokenizer(
            prompt,  
            return_tensors="pt"
        )
        # Move input tensors to the same device as the model
        for key in inputs.keys():
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(self.model.device)
        # Generate the response (return logprobs)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=self.max_new_tokens,
                return_dict_in_generate=True,
                output_logits=True,
            )
            response = self.tokenizer.decode(outputs.sequences[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
        return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt-4.1-mini")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--api_base", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="SEASafeguardDataset", choices=list(AVAILABLE_DATASETS.keys()))
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--cultural", type=str, default="th")
    parser.add_argument("--subset", type=str, default="general")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--prompt_intention", type=str, default=None)
    parser.add_argument("--attack_engine", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--input_path", type=str, default="./inputs/batch.jsonl")
    parser.add_argument("--output_path", type=str, default="./outputs/llms/inference_safeguard.jsonl")
    args = parser.parse_args()

    random.seed(args.seed)

    # Create batch files
    if not os.path.exists(args.input_path):
        dataset = load_dataset(
            args.dataset_name, 
            args.prompt_intention, 
            args.attack_engine, 
            args.language,
            cultural=args.cultural,
            subset=args.subset,
        )
        samples: List[SafeGuardInput] = dataset.as_samples(split=args.dataset_split)
        random.shuffle(samples)

        if args.max_samples is not None:
            samples = samples[:args.max_samples]
            
        # Create a folder to contain the input file
        os.makedirs(os.path.dirname(args.input_path), exist_ok=True)

        for sample_id, sample in enumerate(samples):
            with open(args.input_path, "a") as f:
                f.write(json.dumps({
                    "custom_id": f"request-{sample_id + 1}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": args.model_name,
                        "messages": [
                            {"role": "user", "content": sample.prompt},
                        ],
                        "max_tokens": 2048,
                    },
                }, ensure_ascii=False) + "\n")

    # Initialize the model client
    client = OpenAI(api_key=args.api_key)

    # Upload batch input file to OpenAI
    if not os.path.exists(args.output_path.replace("all_prompts.jsonl", "batch_obj.json")):
        batch_input_file = client.files.create(
            file=open(args.input_path, "rb"),
            purpose="batch"
        )
        print(batch_input_file)

        batch_input_file_id = batch_input_file.id

        batch_obj = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        print(batch_obj)

        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

        with open(args.output_path.replace("all_prompts.jsonl", "batch_obj.json"), "w") as f:
            json.dump(batch_obj.to_dict(), f, indent=4)
    else:
        # Retrieve the batch results
        with open(args.output_path.replace("all_prompts.jsonl", "batch_obj.json"), "r") as f:
            data = json.load(f)
            batch_obj = client.batches.retrieve(data["id"])
            file = client.files.content(batch_obj.output_file_id)
            for line in file.text.split("\n"):
                if line.strip() == "":
                    continue
                data = json.loads(line.strip())
                response = data["response"]["body"]["choices"][0]["message"]["content"]
                print(response)
                print("-" * 100)