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
    "facebook/opt-125m": [f"model.decoder.layers.{layer_idx}.self_attn" for layer_idx in range(24)] +
        [f"model.decoder.layers.{layer_idx}" for layer_idx in range(24)],
}


class LogitLens:
    def __init__(
        self, 
        lm_head: nn.Module,
        tokenizer: AutoTokenizer,
        module_names: Union[List[str], str] = None,
        enable_cache: bool = True,
        max_cache_size: int = 10,
        k: int = 50,
    ):
        self.k = k
        self.lm_head = lm_head
        self.tokenizer = tokenizer
        self.enable_cache = enable_cache
        self.max_cache_size = max_cache_size
        self.module_names = self._resolve_module_names(module_names)

        self.intermediate_logits: Dict[str, Tensor] = {}
        self.intermediate_activations: Dict[str, Tensor] = {}

        self.cached_inputs: List[Tensor] = [] # list of Tensor of shape (seq_len, )
        self.cached_attentions: List[Tensor] = [] # list of Tensor of shape (num_layers, num_heads, seq_len, seq_len)
        self.cached_activations: Dict[str, List[List[float]]] = {}
        self.cached_logits: Dict[str, List[List[Tuple[int, float]]]] = {}

    def _clear_intermediate_logits(self):
        self.intermediate_logits: Dict[str, Tensor] = {}

    def _clear_intermediate_activations(self):
        self.intermediate_activations: Dict[str, Tensor] = {}

    def _clear_attentions(self):
        self.cached_inputs: List[Tensor] = []
        self.cached_attentions: List[Tensor] = []

    def _clear_cache(self):
        self.cached_activations: Dict[str, List[List[float]]] = {}
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
            
            # Update intermediate logits and activations
            self.intermediate_logits[name] = logits
            self.intermediate_activations[name] = activations

            if not self.enable_cache:
                return

            # Update cached_activations
            activations = activations.detach().cpu().clone().tolist()
            if name not in self.cached_activations:
                self.cached_activations[name] = activations
            else:
                self.cached_activations[name].extend(activations)
            # Truncate cached_activations
            if len(self.cached_activations[name]) > self.max_cache_size:
                self.cached_activations[name] = self.cached_activations[name][-self.max_cache_size:]
            
            # Update cached_logits
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            topk_logits = sorted_logits[:, :self.k].detach().cpu().clone().tolist()
            topk_indices = sorted_indices[:, :self.k].detach().cpu().clone().tolist()

            if name not in self.cached_logits:
                self.cached_logits[name] = []
            self.cached_logits[name].extend([list(zip(topk_indices[batch_idx], topk_logits[batch_idx])) for batch_idx in range(len(topk_logits))])
            # Truncate cached_logits
            if len(self.cached_logits[name]) > self.max_cache_size:
                self.cached_logits[name] = self.cached_logits[name][-self.max_cache_size:]
        return hook
    
    def _decode(self, logits: Dict[str, List[List[Tuple[int, float]]]]) -> Dict[str, List[List[Tuple[str, float]]]]:
        decoded_logits: Dict[str, List[List[Tuple[str, float]]]] = {}
        for module_name in logits:
            decoded_logits[module_name] = []
            for sample_logits in logits[module_name]:
                tokens = self.tokenizer.convert_ids_to_tokens([token_id for token_id, _ in sample_logits])
                scores = [score for _, score in sample_logits]
                decoded_logits[module_name].append(list(zip(tokens, scores)))
        return decoded_logits
    
    def register_hooks(self, model):
        for name, module in model.named_modules():
            if name in self.module_names:
                module.register_forward_hook(self._add_hook(name))
    
    def cache_logits(self, logits: Tensor, module_name: str):
        # Update cached_logits
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        topk_logits = sorted_logits[:, :self.k].detach().cpu().clone().tolist()
        topk_indices = sorted_indices[:, :self.k].detach().cpu().clone().tolist()

        if module_name not in self.cached_logits:
            self.cached_logits[module_name] = []
        self.cached_logits[module_name].extend([list(zip(topk_indices[batch_idx], topk_logits[batch_idx])) for batch_idx in range(len(topk_logits))])

    def cache_attentions(self, attentions: List[Tensor], tokens: Tensor):
        """
        Input 
            attentions: list of Tensor of shape (num_samples, num_heads, seq_len, seq_len)
            tokens: Tensor of shape (num_samples, seq_len)
        Output
            cached_attentions: Tensor of shape (num_layers, num_heads, seq_len, seq_len)
        """
        self.cached_inputs.append(tokens.squeeze(0))
        self.cached_attentions.append(torch.stack(attentions, dim=1).squeeze(0))

    def fetch_intermediate_logits(self):
        intermediate_logits = self.intermediate_logits
        self._clear_intermediate_logits()
        self._clear_intermediate_activations()
        return intermediate_logits
    
    def fetch_intermediate_activations(self):
        intermediate_activations = self.intermediate_activations
        self._clear_intermediate_logits()
        self._clear_intermediate_activations()
        return intermediate_activations
    
    def fetch_attentions(self):
        inputs = self.cached_inputs
        attentions = self.cached_attentions
        self._clear_attentions()
        return attentions, inputs
    
    def fetch_cache(
        self, 
        return_tokens: bool = True,
        return_logits: bool = False,
        return_activations: bool = False,
    ) -> Dict[str, Any]:
        cache = {}
        if return_tokens:
            cache["tokens"] = self._decode(self.cached_logits)
        if return_logits:
            cache["logits"] = self.cached_logits
        if return_activations:
            cache["activations"] = self.cached_activations
        self._clear_cache()
        return cache