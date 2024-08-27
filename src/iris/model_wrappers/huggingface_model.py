import torch
import transformers
from typing import List
from iris.data_types import Sample, Response


class HuggfaceInferenceLLM:
    def __init__(self, huggingface_model_name_or_path: str, system_prompt: str = None):
        self.model = transformers.pipeline(
            "text-generation", 
            model=huggingface_model_name_or_path,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        self.model_name = huggingface_model_name_or_path
        self.system_prompt = system_prompt

    def __call__(self, samples: List[Sample]) -> List[Response]:
        responses: List[Response] = []
        for sample in samples:
            if self.system_prompt:
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": sample.instruction},
                ]
            else:
                messages = [{"role": "user", "content": sample.instruction}]
            response = self.model(messages)
            response = Response.from_sample(sample, response, self.model_name)
            responses.append(response)
        return responses