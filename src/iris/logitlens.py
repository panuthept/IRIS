import torch
from torch import nn, Tensor
from typing import List, Dict, Union, Tuple


DEFAULT_MODULE_NAMES = {
    "allenai/wildguard": [f"model.layers.{layer_idx}.self_attn" for layer_idx in range(32)] + 
        [f"model.layers.{layer_idx}.mlp" for layer_idx in range(32)] + \
        [f"model.layers.{layer_idx}" for layer_idx in range(32)]
}


class LogitLens:
    def __init__(
        self, 
        lm_head: nn.Module,
        module_names: Union[List[str], str] = None,
        k: int = 5,
    ):
        self.k = k
        self.lm_head = lm_head
        self.module_names = self.resolve_module_names(module_names)
        self.cached_activations: Dict[str, List[List[int]]] = {}

    def resolve_module_names(self, module_names):
        if module_names is None:
            return []
        if isinstance(module_names, str):
            return DEFAULT_MODULE_NAMES.get(module_names, [])
        return module_names

    def register_hooks(self, model):
        for name, module in model.named_modules():
            if name in self.module_names:
                module.register_forward_hook(self.add_hook(name))

    def extract_activations(self, activations: Union[Tuple[Tensor], Tensor]) -> List[List[int]]:
        if isinstance(activations, tuple):
            activations = activations[0]
        logits = self.lm_head(activations[:, -1])
        return [torch.argsort(logits[batch_idx], descending=True)[:self.k].tolist() for batch_idx in range(logits.size(0))]

    def add_hook(self, name):
        def hook(model, input, output):
            if name not in self.cached_activations:
                self.cached_activations[name] = []
            self.cached_activations[name].extend(self.extract_activations(output))
        return hook
    
    def cache_logits(self, logits: Tensor, module_name: str):
        if module_name not in self.cached_activations:
            self.cached_activations[module_name] = []
        self.cached_activations[module_name].extend([torch.argsort(logits[batch_idx], descending=True)[:self.k].tolist() for batch_idx in range(logits.size(0))])