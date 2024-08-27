import json
from typing import List
from iris.data_types import Response


def save_model_answers(responses: List[Response], path: str):
    with open(path, "w") as f:
        for response in responses:
            f.write(json.dumps(response) + "\n")


def load_model_answers(path) -> List[Response]:
    responses: List[Response] = []
    with open(path, "r") as f:
        for line in f:
            response = json.loads(line)
            responses.append(Response(**response))
    return responses