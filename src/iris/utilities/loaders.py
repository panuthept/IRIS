import json
from typing import List
from iris.data_types import Sample, ModelResponse, IRISConfig, IRISL2Config


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
    if "layer_labels" in config:
        layer_labels = config.pop("layer_labels")
        config["layer_labels"] = {module_name: {int(final_label): layer_label for final_label, layer_label in layer_labels[module_name].items()} for module_name in layer_labels}
    if "layer_weights" in config:
        layer_weights = config.pop("layer_weights")
        config["layer_weights"] = {module_name: {int(final_label): layer_weight for final_label, layer_weight in layer_weights[module_name].items()} for module_name in layer_weights}
    return IRISConfig(**config)


def load_iris_l2_config(path) -> IRISL2Config:
    config = json.load(open(path, "r"))
    if "layer_weights" in config:
        layer_weights = config.pop("layer_weights")
        config["layer_weights"] = {module_name: {int(final_label): layer_weight for final_label, layer_weight in layer_weights[module_name].items()} for module_name in layer_weights}
    return IRISL2Config(**config)