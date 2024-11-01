import os
import sys
import yaml
import json
import numpy as np
from typing import List, Dict
from datasets import load_dataset
from collections import defaultdict
from iris.data_types import Sample
from iris.datasets.base import Dataset
from iris.prompt_template import PromptTemplate


class PromptSourceDataset(Dataset):

    template_paths = {
        "commonsense_qa": "",
        "dream": "dream",
        "quail": "quail",
        "quartz": "quartz",
        "social_i_qa": "social_i_qa",
        "wiqa": "wiqa",
        "cosmos_qa": "cosmos_qa",
        "qasc": "qasc",
        "quarel": "quarel",
        "sciq": "sciq",
        "wiki_hop": "wiki_hop/original",
        "adversarial_qa": ["adversarial_qa/dbidaf", "adversarial_qa/dbert", "adversarial_qa/droberta"],
        "quoref": "quoref",
        "ropes": "ropes",
        "duorc": ["duorc/SelfRC", "duorc/ParaphraseRC"],
        "hotpot_qa": "kilt_tasks/hotpotqa", 
        "wiki_qa": "wiki_qa",
        "amazon_polarity": "amazon_polarity",
        "app_reviews": "app_reviews",
        "imdb": "imdb",
        "rotten_tomatoes": "rotten_tomatoes",
        "yelp_review_full": "yelp_review_full",
        "ag_news": "ag_news",
        "dbpedia_14": "dbpedia_14",
        "trec": "trec",
        "common_gen": "common_gen",
        "wiki_bio": "wiki_bio", 
        "cnn_dailymail": "cnn_dailymail/3.0.0",
        "gigaword": "gigaword",
        "multi_news": "multi_news",
        "samsum": "samsum",
        "xsum": "xsum",
        "mrpc": "glue/mrpc",
        "paws": "paws/labeled_final",
        "qqp": "glue/qqp", 
        "copa": "super_glue/copa",
        "hellaswag": "hellaswag",
        "story_cloze": "story_cloze/2016",
        "anli_r1": "anli",
        "anli_r2": "anli",
        "anli_r3": "anli",
        "cb": "super_glue/cb",
        "rte": "super_glue/rte",
        "wsc": "super_glue/wsc.fixed",
        "winogrande": "winogrande/winogrande_xl",
        "wic": "super_glue/wic",
        }

    huggingface_datasets = {
        "commonsense_qa": "",
        "dream": "dream",
        "quail": "quail",
        "quartz": "quartz",
        "social_i_qa": "social_i_qa",
        "wiqa": "wiqa",
        "cosmos_qa": "cosmos_qa",
        "qasc": "qasc",
        "quarel": "quarel",
        "sciq": "sciq",
        "wiki_hop": "wiki_hop/original",
        "adversarial_qa": ["adversarial_qa/dbidaf", "adversarial_qa/dbert", "adversarial_qa/droberta"],
        "quoref": "quoref",
        "ropes": "ropes",
        "duorc": ["duorc/SelfRC", "duorc/ParaphraseRC"],
        "hotpot_qa": "kilt_tasks/hotpotqa", 
        "wiki_qa": "wiki_qa",
        "amazon_polarity": "amazon_polarity",
        "app_reviews": "app_reviews",
        "imdb": "imdb",
        "rotten_tomatoes": "rotten_tomatoes",
        "yelp_review_full": "yelp_review_full",
        "ag_news": "ag_news",
        "dbpedia_14": "dbpedia_14",
        "trec": "trec",
        "common_gen": "common_gen",
        "wiki_bio": "wiki_bio", 
        "cnn_dailymail": "cnn_dailymail/3.0.0",
        "gigaword": "gigaword",
        "multi_news": "multi_news",
        "samsum": "samsum",
        "xsum": "xsum",
        "mrpc": "glue/mrpc",
        "paws": "paws/labeled_final",
        "qqp": "glue/qqp", 
        "copa": "aps/super_glue",
        "hellaswag": "hellaswag",
        "story_cloze": "LSDSem/story_cloze",
        "anli_r1": "anli",
        "anli_r2": "anli",
        "anli_r3": "anli",
        "cb": "aps/super_glue",
        "rte": "aps/super_glue",
        "wsc": "aps/super_glue",
        "winogrande": "allenai/winogrande",
        "wic": "aps/super_glue",
        }
    
    def __init__(
            self, 
            task_name: str,
            sub_task_name: str = None,
            prompt_name: List[str] = None,
            path: str = "./data/datasets/promptsource/promptsource/templates",
            cache_dir: str = "./data/datasets/promptsource/promptsource/templates",
    ):  
        from promptsource.templates import DatasetTemplates
        
        self.task_name = task_name
        self.prompt_name = prompt_name
        self.path = path
        self.cache_dir = cache_dir

        assert task_name in self.task_available()
        if prompt_name is not None:
            assert prompt_name in self.prompt_name_available(task_name, path)
        
        self.data = self._load_dataset()

    @classmethod
    def task_available(cls) -> List[str]:
        return ["commonsense_qa", "dream", "quail", "quartz", "social_i_qa", "wiqa", "cosmos_qa",
                "qasc", "quarel", "sciq", "wiki_hop", "adversarial_qa", "quoref", "ropes", "duorc",
                "hotpot_qa", "wiki_qa", "amazon_polarity", "app_reviews", "imdb", "rotten_tomatoes",
                "yelp_review_full", "ag_news", "dbpedia_14", "trec", "common_gen", "wiki_bio", 
                "cnn_dailymail", "gigaword", "multi_news", "samsum", "xsum", "mrpc", "paws", "qqp", 
                "anli_r1", "anli_r2", "anli_r3", "cb", "rte", "copa", "hellaswag", "story_cloze", "wsc", "winogrande", "wic",
                ]

    @classmethod
    def prompt_name_available(cls, task_name: str, path: str,) -> List[str]:
        yaml_path = os.path.join(path, cls.template_paths[task_name], "templates.yaml")
        yaml_dict = yaml.load(open(yaml_path, "r"), Loader=yaml.FullLoader)
        prompt_name = [template.name for template in yaml_dict["templates"].values()]
        return prompt_name

    def _get_anli(self) -> Dict[str, List]:
        dataset: Dict[str, Dict] = defaultdict(dict)
        datasets = load_dataset(self.huggingface_datasets[self.task_name], cache_dir=self.cache_dir)
        if "r1" in self.task_name:
            dataset["train"] = datasets["train_r1"]
            dataset["validation"] = datasets["dev_r1"]
            dataset["test"] = datasets["test_r1"]
        elif "r2" in self.task_name:
            dataset["train"] = datasets["train_r2"]
            dataset["validation"] = datasets["dev_r2"]
            dataset["test"] = datasets["test_r2"]
        elif "r3" in self.task_name:
            dataset["train"] = datasets["train_r3"]
            dataset["validation"] = datasets["dev_r3"]
            dataset["test"] = datasets["test_r3"]
        
        return dataset

    def _get_super_glue(self) -> Dict[str, List]:
        if self.task_name == "wsc":
            dataset = load_dataset(self.huggingface_datasets[self.task_name], "wsc.fixed", cache_dir=self.cache_dir)
        else:
            dataset = load_dataset(self.huggingface_datasets[self.task_name], self.task_name, cache_dir=self.cache_dir)
        return dataset

    def _get_story_cloze(self) -> Dict[str, List]:
        dataset = load_dataset(self.huggingface_datasets[self.task_name], "2016", cache_dir=self.cache_dir)
        return dataset

    def _get_winogrande(self) -> [str, List]:
        dataset = load_dataset(self.huggingface_datasets[self.task_name], "winogrande_xl", cache_dir=self.cache_dir)
        return dataset

    def _load_dataset(self) -> Dict[str, List]:
        # Load example from the datasets and prompts
        if "anli" in self.task_name:
            dataset = self._get_anli()
        elif self.task_name in ["cb", "rte", "copa", "wsc", "wic"]:
            dataset = self._get_super_glue()
        elif "story_cloze" in self.task_name:
            dataset = self._get_story_cloze()
        elif "winogrande" in self.task_name:
            dataset = self._get_winogrande()
        else:
            dataset = load_dataset(self.huggingface_datasets[self.task_name], cache_dir=self.cache_dir)

        # Load templates
        if self.prompt_name is None:
            self.prompt_name = self.prompt_name_available(self.task_name, self.path)
        prompts = DatasetTemplates(self.template_paths[self.task_name])

        # Read dataset (original prompt and reference answer)
        datasets: Dict[str, Dict] = defaultdict(dict)
        train_data = []
        if "train" in dataset:
            for prompt_name in self.prompt_name:
                prompt = prompts[prompt_name]
                for example in dataset["train"]:                
                    result = prompt.apply(example)
                    train_data.append({
                        "instructions": [result[0]], 
                        "reference_answers": [result[1]],
                        })
                datasets.update({"train": train_data})
        del train_data
        
        valid_data = []
        if "validation" in dataset:
            for prompt_name in self.prompt_name:
                prompt = prompts[prompt_name]
                for example in dataset["validation"]:                
                    result = prompt.apply(example)
                    valid_data.append({
                        "instructions": [result[0]], 
                        "reference_answers": [result[1]],
                        })
                datasets.update({"validation": valid_data})
        del valid_data
        
        test_data = []
        if "test" in dataset:
            for prompt_name in self.prompt_name:
                prompt = prompts[prompt_name]
                for example in dataset["test"]:                
                    result = prompt.apply(example)
                    test_data.append({
                        "instructions": [result[0]], 
                        "reference_answers": [result[1]],
                        })
                datasets.update({"test": test_data})
        del test_data
        
        return datasets

    def get_size(self, split: str = "test") -> int:
        return len(self.data[split])

    def as_samples(self, split: str = "test", prompt_template: PromptTemplate = None) -> List[Sample]:
        samples: List[Sample] = []
        for sample in self.data[split]:
            samples.append(
                Sample(
                    instructions=sample["instructions"],
                    reference_answers=sample["reference_answers"],
                    prompt_template=PromptTemplate(
                        instruction_template="{instruction}",
                        instruction_answer_template="{instruction}\n\n{answer}",
                    ) if prompt_template is None else prompt_template
                )
            )
        return samples


if __name__ == "__main__":
    dataset = PromptSourceDataset(task_name="winogrande")
    samples = dataset.as_samples(split="test")
    print(f"{samples[0].get_prompts()[0]}")
    print()
    print(f"Label: {samples[0].reference_answers[0]}")
    print("-" * 100)
