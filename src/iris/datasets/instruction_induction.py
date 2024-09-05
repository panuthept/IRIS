import os
import json
import numpy as np
from typing import List, Dict
from iris.data_types import Sample
from iris.datasets.base import Dataset
from iris.prompt_template import PromptTemplate


class InstructionIndutionDataset(Dataset):
    def __init__(
            self, 
            task_name: str,
            in_context_examples_num: int = 0,
            in_context_seed: int = 42,
            path: str = "./data/datasets/instruction_induction",
    ):
        assert task_name in self.task_available()
        self.task_name = task_name
        self.in_context_examples_num = in_context_examples_num
        self.in_context_seed = in_context_seed
        self.data = self._load_dataset(path)

    @classmethod
    def task_available(cls) -> List[str]:
        return ["diff", "sum",  "first_word_letter", "second_word_letter", "rhymes",
                "synonyms", "singular_to_plural", "translation_en-es", "translation_en-fr",
                "translation_en-de", "antonyms", "letters_list", "larger_animal", "num_to_verbal",
                "sentiment", "orthography_starts_with", "common_concept", "informal_to_formal",
                "taxonomy_animal", "active_to_passive", "negation", "word_in_context", 
                "cause_and_effect", "sentence_similarity"]
    
    def _get_datum(
            self, 
            instructions: str, 
            example: str, 
            in_context_examples: List[str] = None
    ):
        if self.task_name in ["cause_and_effect"]:
            datum = {
                "instructions": instructions,
                "query": f"{example['cause']} {example['effect']}",
                "reference_answers": [example["cause"]],
            }
        elif self.task_name in ["common_concept"]:
            datum = {
                "instructions": instructions,
                "query": example["concept"],
                "reference_answers": example["items"],
            }
        elif self.task_name in ["rhymes"]:
            datum = {
                "instructions": instructions,
                "query": example["input"],
                "reference_answers": [answer for answer in example["other_rhymes"] if answer != example["input"]],
            }
        elif self.task_name in ["translation_en-de", "translation_en-es", "translation_en-fr"]:
            datum = {
                "instructions": instructions,
                "query": example["input"],
                "reference_answers": example["possible_translations"],
            }
        elif self.task_name in ["word_in_context"]:
            datum = {
                "instructions": instructions,
                "query": example["input"],
                "reference_answers": example["possible_outputs"],
            }
        else:
            datum = {
                "instructions": instructions,
                "query": example["input"],
                "reference_answers": [example["output"]],
            }

        if in_context_examples is not None:
            datum["examples"] = in_context_examples
        return datum

    def _load_dataset(self, path: str) -> Dict[str, List]:
        annotations_path = os.path.join(path, "data", "annotations")
        execute_path = os.path.join(path, "data", "raw", "execute")
        induce_path = os.path.join(path, "data", "raw", "induce")

        # Load the instruction induction dataset
        with open(f"{annotations_path}/{self.task_name}.json", encoding="utf-8") as f_examples:
            data = json.load(f_examples)
        instructions = data["annotations"]

        # Load train set
        train_data = []
        with open(f'{induce_path}/{self.task_name}.json', 'r', encoding='utf-8') as f:
            data = json.load(f)['examples']
            for example in data.values():
                train_data.append(self._get_datum(instructions, example))

        # Add in-context examples (optional)
        in_context_examples = []
        if self.in_context_examples_num > 0:
            np.random.seed(self.in_context_seed)
            in_context_examples = list(np.random.choice(train_data, size=self.in_context_examples_num, replace=False))
            in_context_examples = [(example["query"], example["reference_answers"][0]) for example in in_context_examples]

        # Load test set
        test_data = []
        with open(f'{execute_path}/{self.task_name}.json', 'r', encoding='utf-8') as f:
            data = json.load(f)['examples']
            for example in data.values():
                test_data.append(self._get_datum(instructions, example, in_context_examples=in_context_examples))

        datasets = {
            "train": train_data,
            "test": test_data,
        }
        return datasets

    def get_size(self) -> int:
        return len(self.data)

    def as_samples(self, split: str = "test", prompt_template: PromptTemplate = None) -> List[Sample]:
        samples: List[Sample] = []
        for sample in self.data[split]:
            samples.append(
                Sample(
                    instructions=sample["instructions"],
                    examples=sample.get("examples", []),
                    query=sample["query"],
                    reference_answers=sample["reference_answers"],
                    prompt_template=PromptTemplate(
                        instruction_query_template="Instruction: {instruction}\nInput: {query}\nOutput: ",
                        query_answer_template="Input: {query}\nOutput: {answer}",
                        instruction_examples_query_template="Instruction: {instruction}\n{examples}\nInput: {query}\nOutput: ",
                    ) if prompt_template is None else prompt_template
                )
            )
        return samples


if __name__ == "__main__":
    dataset = InstructionIndutionDataset(
        task_name="word_in_context",
        in_context_examples_num=5,
    )
    samples = dataset.as_samples(split="test")
    print(f"{samples[0].get_prompts()[0]}")
    print()
    print(f"Label: {samples[0].reference_answers[0]}")
    print("-" * 100)
