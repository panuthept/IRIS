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
    parser.add_argument("--input_path", type=str, default="./outputs")
    parser.add_argument("--ref_path", type=str, default="./outputs/outputs/gemma-2-9b-it")
    parser.add_argument("--model_name", type=str, default="gpt-4.1-mini")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--api_base", type=str, default=None)
    parser.add_argument("--n", type=int, default=3)
    args = parser.parse_args()

    # Initialize the model client
    client = OpenAI(api_key=args.api_key)

    langs = ["en", "in", "ms", "my", "ta", "th", "tl", "vi"]
    suffix_paths = []
    for lang in langs:
        suffix_paths.append(f"SEASafeguardDataset/general/{lang}/test")
        if lang == "en":
            suffix_paths.append(f"SEASafeguardDataset/{lang}_cultural/en/test")
        else:
            suffix_paths.append(f"SEASafeguardDataset/{lang}_cultural/en/test")
            suffix_paths.append(f"SEASafeguardDataset/{lang}_cultural/{lang}/test")
            suffix_paths.append(f"SEASafeguardDataset/{lang}_cultural_handwritten/en/test")
            suffix_paths.append(f"SEASafeguardDataset/{lang}_cultural_handwritten/{lang}/test")

    for suffix_path in suffix_paths:
        # Get reference data
        ref_path = args.ref_path + "/" + suffix_path + "/all_prompts.jsonl"
        with open(ref_path, "r") as ref_file:
            ref_data = []
            for line in ref_file:
                if line.strip() == "":
                    continue
                data = json.loads(line.strip())
                data["responses"] = []
                ref_data.append(data)
        # Iterate through the runs and collect responses
        for i in range(args.n):
            output_path = os.path.join(args.input_path, f"{args.model_name}-run{i+1}")
            print(output_path)
            batch_path = output_path + "/" + suffix_path + "/batch_obj.json"
            # Retrieve the batch results
            with open(batch_path, "r") as f:
                data = json.load(f)
                batch_obj = client.batches.retrieve(data["id"])
                assert batch_obj.status == "completed", f"Batch {batch_obj.id} is not completed. Status: {batch_obj.status}"
                file = client.files.content(batch_obj.output_file_id)
                for sample_id, line in enumerate(file.text.split("\n")):
                    if line.strip() == "":
                        continue
                    data = json.loads(line.strip())
                    response = data["response"]["body"]["choices"][0]["message"]["content"]
                    ref_data[sample_id]["responses"].append(response)
        # Save the collected responses
        output_file = os.path.join(args.input_path, args.model_name, suffix_path + "/all_prompts.jsonl")
        print(output_file)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            for sample in ref_data:
                f.write(json.dumps(sample) + "\n")