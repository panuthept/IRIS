import argparse
from typing import List
from iris.data_types import ModelResponse
from iris.utilities.loaders import load_model_answers
from iris.metrics.exact_match import ExactMatchMetric
from iris.benchmarks import JailbreakBenchBenchmark
from tqdm import tqdm
from iris.model_wrappers.generative_models.huggingface_model import HuggfaceGenerativeLLM
from iris.model_wrappers.generative_models.transformer_lens_model import TransformerLensGenerativeLLM
import torch
from iris.data_types import Sample, ModelResponse


if __name__ == "__main__":

    parser = parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument("--activation_name", type=str, default="attn")
    args = parser.parse_args()

    model = HuggfaceGenerativeLLM(
                args.model_name,
                pipeline_kwargs={
                    "torch_dtype": torch.bfloat16,
                    "model_kwargs": {
                        "cache_dir": "./data/models",
                        "local_files_only": False,
                    }
                },
                cache_path="./cache",
            )
    
    print(f"Device: {model.llm.device}")
    
    benchmark = JailbreakBenchBenchmark()
    # Inference for each task
    evaluation_settings = benchmark.get_evaluation_settings()
    # setting eval
    eval_comp = {}
    missing_files = {}
    
    for setting in tqdm(evaluation_settings, desc="Inference"):
        if setting.get("setting_name", None) in ['Benign (Original)', 'Harmful (Original)']: continue

        # Load the dataset
        dataset = benchmark.get_dataset(
            intention=setting.get("intention", None),
            category=setting.get("category", None),
            attack_engine=setting.get("attack_engine", None),
        )
        
        samples: List[Sample] = dataset.as_samples(split="test", prompt_template=benchmark.prompt_template)

        for layer in range(model.llm.model.config.num_hidden_layers):
            for head in range(model.llm.model.config.num_attention_heads):
                
                direct_path = f"{benchmark.save_path}/{setting['save_name']}/{args.model_name}"
                output_path = f"{direct_path}/{args.activation_name}_L{layer}_H{head}_response.jsonl"
                
                if setting['setting_name'] not in eval_comp.keys(): 
                    eval_comp[setting['setting_name']] = {}
                    missing_files[setting['setting_name']] = []
                if f'L{layer}-H{head}' not in eval_comp[setting['setting_name']].keys(): 
                    eval_comp[setting['setting_name']][f'L{layer}-H{head}'] = []
                
                import os 
                from pathlib import Path
                check_file = Path(output_path)
                    
                if not check_file.is_file():
                    missing_files[setting['setting_name']].append(output_path)
                    continue
                
                responses: List[ModelResponse] = load_model_answers(output_path)
                print(f"Loaded {len(responses)} responses from {output_path}")

                assert len(responses) == len(samples), f'Expected: {len(samples)}, Got: {len(responses)}'

                for idx in range(len(responses)):
                    cur_comp_score = sum(responses[idx].component_scores) / len(responses[idx].component_scores)
                    # eval_comp[setting['setting_name']][f'L{layer}-H{head}'].append(cur_comp_score)
                    # print(f'{idx}: {len(responses[idx].component_scores)}')
                    eval_comp[setting['setting_name']][f'L{layer}-H{head}'].append(cur_comp_score)

                assert len(samples) == len(eval_comp[setting['setting_name']][f'L{layer}-H{head}'])
                cur_eval = sum(eval_comp[setting['setting_name']][f'L{layer}-H{head}']) / len(eval_comp[setting['setting_name']][f'L{layer}-H{head}'])
                eval_comp[setting['setting_name']][f'L{layer}-H{head}'] = cur_eval


    import pickle
    with open('/ist-project/scads/sit/IRIS/missing_exp_paths.pickle', 'wb') as handle:
        pickle.dump(missing_files, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'saving missing file path into pickle')

    with open('/ist-project/scads/sit/IRIS/eval_comp.pickle', 'wb') as handle:
        pickle.dump(eval_comp, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'saving eval component into pickle')
                