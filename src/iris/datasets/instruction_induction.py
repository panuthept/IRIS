import os
import json
from typing import List, Dict
from iris.data_types import Sample
from collections import defaultdict
from iris.datasets.base import Dataset


class InstructionIndutionDataset(Dataset):
    instruction_paths = [
        "diff.json",
        "sum.json",
        "first_word_letter.json",
        "second_word_letter.json",
        "rhymes.json",
        "synonyms.json",
        "singular_to_plural.json",
        "translation_en-es.json",
        "translation_en-fr.json",
        "translation_en-de.json",
        "antonyms.json",
        "letters_list.json",
        "larger_animal.json",
        "num_to_verbal.json",
        "sentiment.json",
        "orthography_starts_with.json",
        "common_concept.json",
        "informal_to_formal.json",
        "taxonomy_animal.json",
        "active_to_passive.json",
        "negation.json",
        "word_in_context.json",
        "cause_and_effect.json",
        "sentence_similarity.json"
    ]

    def __init__(
            self, 
            task_name: str,
            path: str = "./data/datasets/instruction_induction",
    ):
        assert task_name in self.task_available()
        self.task_name = task_name
        self.data = self._load_dataset(path)

    def task_available(self) -> List[str]:
        return ["diff", "sum",  "first_word_letter.json", "second_word_letter", "rhymes",
                "synonyms", "singular_to_plural", "translation_en-es", "translation_en-fr",
                "translation_en-de", "antonyms", "letters_list", "larger_animal", "num_to_verbal",
                "sentiment", "orthography_starts_with", "common_concept", "informal_to_formal",
                "taxonomy_animal", "active_to_passive", "negation", "word_in_context", 
                "cause_and_effect", "sentence_similarity"]

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
        samples = defaultdict(dict)
        for id_, example in test_data.items():
            samples[int(id_)] = {
                "instructions": instructions,
                "query": example["input"],
                "reference_answers": [example["output"]],
            }

        samples = list(samples.values())
        return samples

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
                        "instruction_with_query": "Instruction: {instruction}\nInput: {query}\n",
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
