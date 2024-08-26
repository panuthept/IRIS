import json
from typing import List
from iris.data_types import Sample


def load_model_answers() -> List[Sample]:
    """Load the model answers from the model_answers.json file."""
    with open("model_answers.json", "r") as f:
        model_answers = json.load(f)

    return [Sample(**sample) for sample in model_answers]