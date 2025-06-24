import os
import json
import torch
import random
import argparse
from tqdm import tqdm
from openai import OpenAI
from typing import List, Dict
from iris.data_types import SafeGuardInput
from iris.datasets import load_dataset, AVAILABLE_DATASETS
from transformers import AutoModelForCausalLM, AutoTokenizer


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
            )
            response = outputs.choices[0].message.content
        else:
            outputs = self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
            )
            response = outputs.choices[0].text
        return response
    
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
    parser.add_argument("--model_name", type=str, default="google/gemma-3-27b-it")
    parser.add_argument("--tokenizer_name", type=str, default="google/gemma-3-27b-it")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--api_base", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="SEASafeguardDataset", choices=list(AVAILABLE_DATASETS.keys()))
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--cultural", type=str, default="th")
    parser.add_argument("--subset", type=str, default="general")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--prompt_intention", type=str, default=None)
    parser.add_argument("--attack_engine", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--output_path", type=str, default="./outputs/llms/inference_safeguard.jsonl")
    args = parser.parse_args()

    random.seed(args.seed)

    # Initial model
    if args.api_key is not None and args.api_base is not None:
        # Use OpenAI API
        llm = APIModel(
            model_name=args.model_name,
            api_key=args.api_key,
            api_base=args.api_base,
        )
    else:
        # Use Hugging Face API
        llm = HFModel(
            model_name=args.model_name,
            tokenizer_name=args.tokenizer_name,
        )

    # Initial dataset
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

    # Create save directory
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    with open(args.output_path, "w") as f:
        for sample in tqdm(samples):
            # Get responses
            responses = []
            for _ in range(args.n):
                response = llm.complete(sample.prompt)
                responses.append(response)
            # Save results
            f.write(json.dumps({
                "prompt": sample.prompt,
                "prompt_gold_label": sample.prompt_gold_label,
                "responses": responses,
            }, ensure_ascii=False) + "\n")