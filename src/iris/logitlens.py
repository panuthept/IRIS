import torch
from torch import nn, Tensor
from transformers import AutoTokenizer
from typing import List, Dict, Union, Tuple


DEFAULT_MODULE_NAMES = {
    "allenai/wildguard": [f"model.layers.{layer_idx}.self_attn" for layer_idx in range(32)] + 
        [f"model.layers.{layer_idx}.mlp" for layer_idx in range(32)] + \
        [f"model.layers.{layer_idx}" for layer_idx in range(32)],
    "mistralai/Mistral-7B-v0.3": [f"model.layers.{layer_idx}.self_attn" for layer_idx in range(32)] + 
        [f"model.layers.{layer_idx}.mlp" for layer_idx in range(32)] + \
        [f"model.layers.{layer_idx}" for layer_idx in range(32)],
}


class LogitLens:
    def __init__(
        self, 
        lm_head: nn.Module,
        tokenizer: AutoTokenizer,
        module_names: Union[List[str], str] = None,
        k: int = 5,
    ):
        self.k = k
        self.lm_head = lm_head
        self.tokenizer = tokenizer
        self.module_names = self._resolve_module_names(module_names)
        self.cached_activations: Dict[str, List[List[int]]] = {}

    def clear_cache(self):
        self.cached_activations: Dict[str, List[List[int]]] = {}

    def _resolve_module_names(self, module_names):
        if module_names is None:
            return []
        if isinstance(module_names, str):
            return DEFAULT_MODULE_NAMES.get(module_names, [])
        return module_names

    def _extract_activations(self, activations: Union[Tuple[Tensor], Tensor]) -> List[List[int]]:
        if isinstance(activations, tuple):
            activations = activations[0]
        logits = self.lm_head(activations[:, -1])
        return [torch.argsort(logits[batch_idx], descending=True)[:self.k].tolist() for batch_idx in range(logits.size(0))]

    def _add_hook(self, name):
        def hook(model, input, output):
            if name not in self.cached_activations:
                self.cached_activations[name] = []
            self.cached_activations[name].extend(self._extract_activations(output))
        return hook
    
    def register_hooks(self, model):
        for name, module in model.named_modules():
            if name in self.module_names:
                module.register_forward_hook(self._add_hook(name))
    
    def cache_logits(self, logits: Tensor, module_name: str):
        if module_name not in self.cached_activations:
            self.cached_activations[module_name] = []
        self.cached_activations[module_name].extend([torch.argsort(logits[batch_idx], descending=True)[:self.k].tolist() for batch_idx in range(logits.size(0))])

    def get_last_activations(self) -> Dict[str, List[str]]:
        last_activations = {}
        for module_name in self.cached_activations:
            last_activations[module_name] = self.tokenizer.convert_ids_to_tokens(self.cached_activations[module_name][-1])
        return last_activations
    
    def get_activations(self) -> Dict[str, List[Tuple[str, int]]]:
        # Accumulate activations
        accum_activations = {}
        for module_name in self.cached_activations:
            accum_activations[module_name] = {}
            for sample_activations in self.cached_activations[module_name]:
                for token_id in sample_activations:
                    token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
                    if token not in accum_activations[module_name]:
                        accum_activations[module_name][token] = 0
                    accum_activations[module_name][token] += 1
            # Sort activations
            accum_activations[module_name] = sorted(accum_activations[module_name].items(), key=lambda x: x[1], reverse=True)[:self.k]
        return accum_activations