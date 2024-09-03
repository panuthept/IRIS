import os
import json
from typing import List, Dict
from iris.data_types import Sample
from collections import defaultdict
from iris.datasets.base import Dataset


class InstructionIndutionDataset(Dataset):
    def __init__(
            self, 
            task_name: str,
            path: str = "./data/datasets/instruction_induction",
    ):
        assert task_name in self.task_available()
        self.task_name = task_name
        self.data = self._load_dataset(path)

    @classmethod
    def task_available(cls) -> List[str]:
        return ["diff", "sum",  "first_word_letter", "second_word_letter", "rhymes",
                "synonyms", "singular_to_plural", "translation_en-es", "translation_en-fr",
                "translation_en-de", "antonyms", "letters_list", "larger_animal", "num_to_verbal",
                "sentiment", "orthography_starts_with", "common_concept", "informal_to_formal",
                "taxonomy_animal", "active_to_passive", "negation", "word_in_context", 
                "cause_and_effect", "sentence_similarity"]
    
    def _get_datum(self, instructions, example):
        if self.task_name in ["cause_and_effect"]:
            return {
                "instructions": instructions,
                "query": example["cause"],
                "reference_answers": [example["effect"]],
            }
        elif self.task_name in ["common_concept"]:
            return {
                "instructions": instructions,
                "query": example["concept"],
                "reference_answers": example["items"],
            }
        elif self.task_name in ["rhymes"]:
            return {
                "instructions": instructions,
                "query": example["input"],
                "reference_answers": example["other_rhymes"],
            }
        elif self.task_name in ["translation_en-de", "translation_en-es", "translation_en-fr"]:
            return {
                "instructions": instructions,
                "query": example["input"],
                "reference_answers": example["possible_translations"],
            }
        elif self.task_name in ["word_in_context"]:
            return {
                "instructions": instructions,
                "query": example["input"],
                "reference_answers": example["possible_outputs"],
            }
        else:
            return {
                "instructions": instructions,
                "query": example["input"],
                "reference_answers": [example["output"]],
            }

    def _load_dataset(self, path: str) -> Dict[str, List]:
        annotations_path = os.path.join(path, "data", "annotations")
        execute_path = os.path.join(path, "data", "raw", "execute")

        # Load the instruction induction dataset
        with open(f"{annotations_path}/{self.task_name}.json", encoding="utf-8") as f_examples:
            data = json.load(f_examples)
        instructions = data["annotations"]

        # Load reference answer dataset
        with open(f'{execute_path}/{self.task_name}.json', 'r', encoding='utf-8') as test_f:
            test_data = json.load(test_f)['examples']

        # Get the original instruction induction and reference answer
        data = defaultdict(dict)
        for idx, example in test_data.items():
            data[int(idx)] = self._get_datum(instructions, example)

        data = list(data.values())
        return data

    def get_size(self) -> int:
        return len(self.data)

    def as_samples(self, prompt_template: Dict[str, str] = None) -> List[Sample]:
        samples: List[Sample] = []
        for sample in self.data:
            samples.append(
                Sample(
                    instructions=sample["instructions"],
                    query=sample["query"],
                    reference_answers=sample["reference_answers"],
                    prompt_template={
                        "instruction_with_query": "Instruction: {instruction}\nInput: {query}\nOutput: ",
                    } if prompt_template is None else prompt_template
                )
            )
        return samples


if __name__ == "__main__":
    dataset = InstructionIndutionDataset(task_name="sentiment")
    samples = dataset.as_samples()
    print("InstructionIndutionDataset:")
    print(f"{samples[0].get_prompts()}")
    print(f"Label: {samples[0].reference_answers[0]}")
    print("-" * 100)
