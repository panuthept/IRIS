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

    @abstractmethod
    def _apply_safeguard_template(self, prompt: str, response: Optional[str] = None) -> str:
        raise NotImplementedError
    
    @abstractmethod
    def _complete(self, instruction: str, **kwargs) -> str:
        raise NotImplementedError
    
    def complete(self, instruction: str, **kwargs) -> Dict[str, List[Tuple[str, float, float]]]:
        outputs, response = self._complete(instruction, **kwargs)
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
            "instruction": instruction,
            "response": response,
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
        metadata["prompt_instruction"] = prompt_clf["instruction"]
        metadata["prompt_response"] = prompt_clf["response"]
        # Response classification
        response_labels = None
        if response is not None:
            response_clf: Dict[str, List[Tuple[str, float, float]]] = self._response_classify(prompt, response, **kwargs)
            response_labels = response_clf["pred_labels"]
            metadata["response_tokens"] = response_clf["pred_tokens"]
            metadata["response_instruction"] = response_clf["instruction"]
            metadata["response_response"] = response_clf["response"]
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