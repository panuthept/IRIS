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
    
    def __init__(
            self, 
            task_name: str,
            sub_task_name: str = None,
            prompt_name: str = None,
            path: str = "./data/datasets/promptsource/promptsource/templates",
            cache_dir: str = None,
    ):
        from promptsource.templates import DatasetTemplates
        
        self.task_name = task_name
        self.sub_task_name = sub_task_name
        self.prompt_name = prompt_name
        self.path = path
        self.cache_dir = cache_dir

        assert task_name in self.task_available()
        if sub_task_name is not None:
            assert sub_task_name in self.sub_task_available(task_name)
        if prompt_name is not None:
            assert prompt_name in self.prompt_name_available(task_name, sub_task_name, path)
        
        self.data = self._load_dataset()

    @classmethod
    def task_available(cls) -> List[str]:
        return ["acronym_identification", "ade_corpus_v2", "adversarial_qa", "aeslc",
                "ag_news", "ai2_arc", "amazon_polarity", "amazon_reviews_multi", 
                "amazon_us_reviews", "ambig_qa", "anli", "app_reviews", "aqua_rat",
                "art", "asnq", "asset", "banking77", "billsum", "bing_coronavirus_query_set",
                "biosses", "blbooksgenre", "blended_skill_talk", "cbt", "cc_news", "circa",
                "climate_fever", "cnn_dailymail", "codah", "code_x_glue_tc_text_to_code",
                "common_gen", "commonsense_qa", "conv_ai", "conv_ai_2", "conv_ai_3", "coqa",
                "cord19", "cos_e", "cosmos_qa", "covid_qa_castorini", "craffel",
                "craigslist_bargains", "crows_pairs", "dbpedia_14", "discofuse", "discovery",
                "docred", "dream", "drop", "duorc", "e2e_nlg_cleaned", "ecthr_cases", "emo",
                "emotion", "enriched_web_nlg", "esnli", "evidence_infer_treatment", "fever",
                "financial_phrasebank", "freebase_qa", "generated_reviews_enth", "gigaword",
                "glue", "google_wellformed_query", "great_code", "guardian_authorship",
                "gutenberg_time", "hans", "hate_speech18", "head_qa", "health_fact",
                "hellaswag", "hlgd", "hotpot_qa", "humicroedit", "hyperpartisan_news_detection",
                "imdb", "jfleg", "jigsaw_unintended_bias", "kelm", "kilt_tasks", "lama",
                "lambada", "liar", "limit", "math_dataset", "math_qa", "mc_taco", "mdd",
                "medal", "medical_questions_pairs", "meta_woz", "mocha", "movie_rationales",
                "multi_news", "multi_nli", "multi_x_science_sum", "mwsc", "narrativeqa",
                "ncbi_disease", "neural_code_search", "newspop", "nlu_evaluation_data",
                "nq_open", "numer_sense", "onestop_english", "openai_humaneval", "openbookqa",
                "paws", "paws-x", "piqa", "poem_sentiment", "pubmed_qa", "qasc", "qa_srl",
                "qa_zre", "qed", "quac", "quail", "quarel", "quartz", "quora", "quoref",
                "race", "riddle_sense", "ropes", "ropes", "rotten_tomatoes", "samsum",
                "scan", "scicite", "scientific_papers", "sciq", "scitail", "scitldr",
                "selqa", "sem_eval_2010_task_8", "sem_eval_2014_task_1", "sent_comp",
                "sick", "sms_spam", "snips_built_in_intents", "snli", "social_i_qa",
                "species_800", "squad", "squad_adversarial", "squadshifts", "squad_v2", 
                "sst", "story_cloze", "stsb_multi_mt", "subjqa", "super_glue", "swag",
                "tab_fact", "tmu_gfm_dataset", "trec", "trivia_qa", "turk", "tweet_eval",
                "tydiqa", "web_questions", "wiki_bio", "wiki_hop", "wiki_qa", "wiki_split",
                "wino_bias", "winograd_wsc", "winogrande", "wiqa", "xnli", "xquad",
                "xquad_r", "xsum", "yahoo_answers_qa", "yahoo_answers_topics", "yelp_polarity",
                "yelp_review_full", "Zaid", "zest"]

    @classmethod
    def sub_task_available(cls, task_name: str) -> List[str]:
        sub_task_dict = {
            "ade_corpus_v2": ["Ade_corpus_v2_classification", "Ade_corpus_v2_drug_ade_relation", 
                              "Ade_corpus_v2_drug_dosage_relation"],
            "adversarial_qa": ["adversarialQA", "dbert", "dbidaf", "droberta"],
            "ai2_arc": ["ARC-Challenge", "ARC-Easy"],
            "asset": ["ratings", "simplification"],
            "cbt": ["CN", "NE", "P", "V", "raw"],
            "codah": ["codah", "fold_0", "fold_1", "fold_2", "fold_3", "fold_4"],
            "cos_e": ["v1.0", "v1.11"],
            "discofuse": ["discofuse-sport", "discofuse-wikipedia"],
            "duorc": ["ParaphraseRC", "SelfRC"],
            "evidence_infer_treatment": ["1.1", "2.0"],
            "fever": ["v1.0", "v2.0"],
            "glue": ["ax", "cola", "mnli", "mnli_matched", "mnli_mismatched", "mrpc", "qnli",
                     "qqp", "rte", "sst2", "stsb", "wnli"],
            "guardian_authorship": ["cross_genre_1", "cross_topic_1", "cross_topic_4", "cross_topic_7"],
            "hotpot_qa": ["distractor", "fullwiki"],
            "humicroedit": ["subtask-1", "subtask-2"],
            "hyperpartisan_news_detection": ["byarticle", "bypublisher"],
            "kilt_tasks": ["hotpotqa", "nq"],
            "math_dataset": ["algebra__linear_1d", "algebra__linear_1d_composed", "algebra__linear_2d",
                             "algebra__linear_2d_composed"],
            "mdd": ["task1_qa", "task2_recs", "task3_qarecs"],
            "openbookqa": ["additional", "main"],      
            "paws": ["labeled_final", "labeled_swap", "unlabeled_final"],
            "race": ["all", "high", "middle"],
            "scan": ["addprim_jump", "addprim_turn_left", "filler_num0", "filler_num1", "filler_num2",
                     "filler_num3", "length", "simple", "template_around_right", "template_jump_around_right",
                     "template_opposite_right", "template_right"],
            "scientific_papers": ["arxiv", "pubmed"],
            "scitail": ["snli_format", "tsv_format"],
            "squadshifts": ["amazon", "new_wiki", "nyt"],
            "subjqa": ["books", "electronics", "grocery", "movies", "restaurants", "tripadvisor"],
            "super_glue": ["axb", "axg", "boolq", "cb", "copa", "multirc", "record", "rte", "wic", "wsc.fixed"],
            "tweet_eval": ["emoji", "emotion", "hate", "irony", "offensive", "sentiment", "stance_abortion",
                           "stance_atheism", "stance_climate", "stance_feminist", "stance_hillary"],
            "tydiqa": ["primary_task", "secondary_task"],
            "wiki_hop": ["masked", "original"],
            "wino_bias": ["type1_anti", "type1_pro", "type2_anti", "type2_pro"],
            "winograd_wsc": ["wsc273", "wsc285"],
            "winogrande": ["winogrande_debiased", "winogrande_l", "winogrande_m", "winogrande_s",
                           "winogrande_xl", "winogrande_xs"],
            "Zaid": ["coqa_expanded", "quac_expanded"],
        }
        return sub_task_dict.get(task_name)

    @classmethod
    def prompt_name_available(cls, task_name: str, sub_task_name: str, path: str,) -> List[str]:
        if sub_task_name is not None:
            yaml_path = os.path.join(path, task_name, sub_task_name, "templates.yaml")
        else:
            yaml_path = os.path.join(path, task_name, "templates.yaml")

        yaml_dict = yaml.load(open(yaml_path, "r"), Loader=yaml.FullLoader)
        prompt_name = [template.name for template in yaml_dict["templates"].values()]
        return prompt_name

    @classmethod
    def split_available(cls) -> List[str]:
        return self.data.keys()

    def _load_dataset(self) -> Dict[str, List]:
        # Load example from the datasets and prompts
        if self.sub_task_name is not None:
            dataset = load_dataset(f"{self.task_name}/{self.sub_task_name}", cache_dir=self.cache_dir) 
            prompts = DatasetTemplates(f"{self.task_name}/{self.sub_task_name}")
        else:
            dataset = load_dataset(self.task_name, cache_dir=self.cache_dir)
            prompts = DatasetTemplates(self.task_name)
        
        # Select a prompt by its name
        prompt = prompts[self.prompt_name]

        # Read dataset (original prompt and reference answer)
        datasets: Dict[str, Dict] = defaultdict(dict)
        if "train" in dataset:
            train_data = []
            for example in dataset["train"]:                
                result = prompt.apply(example)
                train_data.append({
                    "instructions": [result[0]], 
                    "reference_answers": [result[1]]
                    })
            datasets.update({"train": train_data})
            del train_data
        if "validation" in dataset:
            valid_data = []
            for example in dataset["validation"]:
                result = prompt.apply(example)
                valid_data.append({
                "instructions": [result[0]],
                "reference_answers": [result[1]]
                })
            datasets.update({"validation": valid_data})
            del valid_data
        if "test" in dataset:
            test_data = []
            for example in dataset["test"]:
                result = prompt.apply(example)
                test_data.append({
                "instructions": [result[0]],
                "reference_answers": [result[1]]
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
    dataset = PromptSourceDataset(
                task_name="ag_news",
                prompt_name="classify_question_first")
    samples = dataset.as_samples(split="train")
    print(f"{samples[0].get_prompts()[0]}")
    print()
    print(f"Label: {samples[0].reference_answers[0]}")
    print("-" * 100)
