import torch
from torch import nn, Tensor
from transformers import AutoTokenizer
from typing import List, Dict, Any, Union, Tuple


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
        k: int = 50,
    ):
        self.k = k
        self.lm_head = lm_head
        self.tokenizer = tokenizer
        self.module_names = self._resolve_module_names(module_names)

        self.cached_activations: Dict[str, Tensor] = {}
        self.cached_logits: Dict[str, List[List[Tuple[int, float]]]] = {}

    def _clear_cache(self):
        self.cached_activations: Dict[str, Tensor] = {}
        self.cached_logits: Dict[str, List[List[Tuple[int, float]]]] = {}

    def _resolve_module_names(self, module_names):
        if module_names is None:
            return []
        if isinstance(module_names, str):
            return DEFAULT_MODULE_NAMES.get(module_names, [])
        return module_names

    def _add_hook(self, name):
        def hook(model, input, output):
            # Output of a self-attention layer is a tuple
            if isinstance(output, tuple):
                output = output[0]
            # Get last token activations
            activations = output[:, -1]  # (batch_size, hidden_size)
            logits = self.lm_head(activations)  # (batch_size, vocab_size)
            token_ids: List[List[int]] = [torch.argsort(logits[batch_idx], descending=True)[:self.k].tolist() for batch_idx in range(logits.size(0))]

            # Update cached_activations
            activations = activations.detach().cpu().clone()
            if name not in self.cached_activations:
                self.cached_activations[name] = activations
            else:
                self.cached_activations[name] = torch.cat([self.cached_activations[name], activations], dim=0)
            
            # Update cached_logits
            logits = logits.detach().cpu().clone()
            if name not in self.cached_logits:
                self.cached_logits[name] = []
            self.cached_logits[name].extend(list(zip(token_ids, logits)))
        return hook
    
    def _decode(self, logits: List[List[Tuple[int, float]]]) -> List[List[Tuple[str, float]]]:
        tokens: List[List[Tuple[str, float]]] = []
        for sample_logits in logits:
            tokens = self.tokenizer.convert_ids_to_tokens([token_id for token_id, _ in sample_logits])
            scores = [score for _, score in sample_logits]
            tokens.append(list(zip(tokens, scores)))
        return tokens
    
    def register_hooks(self, model):
        for name, module in model.named_modules():
            if name in self.module_names:
                module.register_forward_hook(self._add_hook(name))
    
    def cache_logits(self, logits: Tensor, module_name: str):
        token_ids: List[List[int]] = [torch.argsort(logits[batch_idx], descending=True)[:self.k].tolist() for batch_idx in range(logits.size(0))]
        # Update cached_logits
        logits = logits.detach().cpu().clone()
        if module_name not in self.cached_logits:
            self.cached_logits[module_name] = []
        self.cached_logits[module_name].extend(list(zip(token_ids, logits)))

    def fetch_cache(self, decode: bool = True) -> Dict[str, Any]:
        cache = {
            "tokens": self._decode(self.cached_logits) if decode else None, 
            "logits": self.cached_logits, 
            "activations": self.cached_activations
        }
        self._clear_cache()
        return cache

    # def get_last_activations(
    #     self, 
    #     decode_activations: bool = True,
    # ) -> Dict[str, Union[List[str], Tensor]]:
    #     last_activations = {}
    #     for module_name in self.cached_logits:
    #         activations = self.cached_logits[module_name][-1]
    #         if decode_activations:
    #             activations = self.tokenizer.convert_ids_to_tokens(activations)
    #         last_activations[module_name] = activations
    #     return last_activations
    
    # def get_activations(self, decode_activations: bool = True) -> Dict[str, Union[List[Tuple[str, int]], Tensor]]:
    #     # Accumulate activations
    #     accum_activations = {}
    #     for module_name in self.cached_logits:
    #         accum_activations[module_name] = {}
    #         for sample_activations in self.cached_logits[module_name]:
    #             for token_id in sample_activations:
    #                 token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
    #                 if token not in accum_activations[module_name]:
    #                     accum_activations[module_name][token] = 0
    #                 accum_activations[module_name][token] += 1
    #         # Sort activations
    #         accum_activations[module_name] = sorted(accum_activations[module_name].items(), key=lambda x: x[1], reverse=True)[:self.k]
    #     return accum_activations