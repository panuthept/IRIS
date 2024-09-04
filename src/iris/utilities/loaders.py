import json
from typing import List
from iris.data_types import ModelResponse


def save_model_answers(responses: List[ModelResponse], path: str):
    with open(path, "w") as f:
        for response in responses:
            f.write(json.dumps(response.as_dict()) + "\n")


def load_model_answers(path) -> List[ModelResponse]:
    responses: List[ModelResponse] = []
    with open(path, "r") as f:
        for line in f:
            response = json.loads(line)
            responses.append(ModelResponse(**{k: v for k, v in response.items() if k in ModelResponse.__dataclass_fields__}))
    return responses