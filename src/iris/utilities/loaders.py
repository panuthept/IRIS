import json
from typing import List
from iris.data_types import Sample, ModelResponse, IRISConfig


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


def load_iris_config(path) -> IRISConfig:
    config = json.load(open(path, "r"))
    intermediate_labels = config.pop("intermediate_labels")
    # Ensure that the intermediate_labels are in the correct format and type
    config["intermediate_labels"] = {module_name: {int(final_label): (intermediate_label, weight) for final_label, (intermediate_label, weight) in intermediate_labels[module_name].items()} for module_name in intermediate_labels}
    return IRISConfig(**config)