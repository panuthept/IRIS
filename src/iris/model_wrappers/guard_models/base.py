import numpy as np
from tqdm import tqdm
from copy import deepcopy
from abc import abstractmethod
from iris.model_wrappers import LLM
from typing import List, Dict, Tuple, Optional
from iris.data_types import Sample, ModelResponse, SafeGuardInput, SafeGuardResponse


class GuardLLM(LLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.valid_tokens = {} # Dictionary of {String: "Safe", String: "Harmful"}

    def prompt_classify(self, sample: Sample) -> Sample:
        sample = deepcopy(sample)
        sample.instructions_pred_label = [self._prompt_classify(inst) for inst in sample.instructions]
        return sample

    def response_classify(self, response: ModelResponse) -> ModelResponse:
        response = deepcopy(response)
        response.answers_pred_label = [self._response_classify(inst, ans) for inst, ans in zip(response.instructions, response.answers)]
        return response
    
    def prompt_classify_batch(self, samples: List[Sample], verbose: bool = True) -> List[Sample]:
        return [self.prompt_classify(sample) for sample in tqdm(samples, disable=not verbose)]
    
    def response_classify_batch(self, responses: List[ModelResponse], verbose: bool = True) -> List[ModelResponse]:
        return [self.response_classify(response) for response in tqdm(responses, disable=not verbose)]
    
    def generate(self, *args, **kwargs):
        return self._prompt_classify(*args, **kwargs)
    
    @abstractmethod
    def _apply_safeguard_template(self, prompt: str, response: Optional[str] = None) -> str:
        raise NotImplementedError
    
    @abstractmethod
    def _complete(self, instruction: str, **kwargs) -> str:
        raise NotImplementedError
    
    def complete(self, instruction: str, **kwargs) -> Dict[str, List[Tuple[str, float, float]]]:
        outputs = self._complete(instruction, **kwargs)
        if outputs is None:
            outputs = [[(valid_token, 0.0, 0.0) for valid_token in self.valid_tokens.keys()]]
        output = outputs[0]
        # Get all token outputs
        tokens = [token for token, _, _ in output]
        tokens_logprobs = [logprob for _, logprob, _ in output]
        tokens_logits = [logit for _, _, logit in output]
        tokens_probs = np.exp(tokens_logprobs) / np.sum(np.exp(tokens_logprobs))
        # Filter out invalid tokens
        output = [(self.valid_tokens[token], logprob, logit) for token, logprob, logit in output if token in self.valid_tokens]
        # Convert logprobs to probabilities
        labels = [label for label, _, _ in output]
        labels_logprobs = [logprob for _, logprob, _ in output]
        labels_logits = [logit for _, _, logit in output]
        labels_probs = np.exp(labels_logprobs) / np.sum(np.exp(labels_logprobs))
        return {
            "pred_labels": list(zip(labels, labels_probs, labels_logits)),
            "pred_tokens": list(zip(tokens, tokens_probs, tokens_logits)),
        }
    
    def _prompt_classify(self, prompt: str, **kwargs) -> Dict[str, List[Tuple[str, float, float]]]:
        instruction: str = self._apply_safeguard_template(prompt=prompt)
        return self.complete(instruction, **kwargs)
    
    def _response_classify(self, prompt: str, response: str, **kwargs) -> Dict[str, List[Tuple[str, float, float]]]:
        instruction: str = self._apply_safeguard_template(prompt=prompt, response=response)
        return self.complete(instruction, **kwargs)
    
    def predict(
        self,
        input: Optional[SafeGuardInput] = None,
        prompt: Optional[str] = None,
        response: Optional[str] = None,
        **kwargs,
    ) -> SafeGuardResponse:
        prompt_gold_label = None
        response_gold_label = None
        if input is not None:
            prompt = input.prompt
            response = input.response
            prompt_gold_label = input.prompt_gold_label
            response_gold_label = input.response_gold_label
        assert prompt is not None, "Prompt cannot be None"
        # Initial metadata
        metadata = {}
        # Prompt classification
        prompt_clf: Dict[str, List[Tuple[str, float, float]]] = self._prompt_classify(prompt, **kwargs)
        prompt_labels = prompt_clf["pred_labels"]
        metadata["prompt_tokens"] = prompt_clf["pred_tokens"]
        # Response classification
        response_labels = None
        if response is not None:
            response_clf: Dict[str, List[Tuple[str, float, float]]] = self._response_classify(prompt, response, **kwargs)
            response_labels = response_clf["pred_labels"]
            metadata["response_tokens"] = response_clf["pred_tokens"]
        # Output formatting
        output = SafeGuardResponse(
            prompt=prompt, 
            response=response,
            prompt_gold_label=prompt_gold_label,
            response_gold_label=response_gold_label,
            prompt_labels=prompt_labels,
            response_labels=response_labels,
            metadata=metadata,
        )
        return output

    @abstractmethod
    def _prompt_classification_tokenizer(self, prompt: str) -> dict:
        raise NotImplementedError
    
    @abstractmethod
    def _response_classification_tokenizer(self, prompt: str, response: str) -> dict:
        raise NotImplementedError

    def tokenize(self, prompt: str, prompt_label: Optional[str] = None, response: Optional[str] = None):
        if prompt_label is not None and response is not None:
            return self._response_classification_tokenizer(prompt, prompt_label, response)
        return self._prompt_classification_tokenizer(prompt)