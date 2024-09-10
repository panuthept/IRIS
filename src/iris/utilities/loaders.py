import json
from typing import List
from iris.data_types import Sample, ModelResponse


def save_samples(samples: List[Sample], path: str):
    with open(path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample.as_dict()) + "\n")


def save_responses(responses: List[ModelResponse], path: str):
    with open(path, "w") as f:
        for response in responses:
            f.write(json.dumps(response.as_dict()) + "\n")


def load_samples(path) -> List[Sample]:
    samples: List[Sample] = []
    with open(path, "r") as f:
        for line in f:
            data = json.loads(line)
            samples.append(Sample.from_dict(data))
    return samples


def load_responses(path) -> List[ModelResponse]:
    responses: List[ModelResponse] = []
    with open(path, "r") as f:
        for line in f:
            data = json.loads(line)
            responses.append(ModelResponse.from_dict(data))
    return responses