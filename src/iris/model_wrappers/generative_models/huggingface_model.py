import os
import torch
import numpy as np
import transformer_lens.utils as utils

from tqdm import tqdm
from typing import Dict
# from peft import LoraConfig
from torch import nn, Tensor
from datasets import Dataset
from jaxtyping import Int, Float
from accelerate import PartialState
from iris.logitlens import LogitLens
from typing import List, Callable, Union, Tuple, Optional
from iris.model_wrappers.generative_models.base import GenerativeLLM
# from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from iris.data_types import IRISConfig, IRISL2Config, IRISDiffTripletConfig, IRISCLConfig


# class IRISTrainer(SFTTrainer):
#     """
#     Example of intermediate_labels:
#     intermediate_labels = {
#         "model.layers.18": {label_id: token_id},
#         "model.layers.19": {label_id: token_id},
#         ...
#     }
#     """
#     def __init__(
#         self, 
#         logitlens: LogitLens, 
#         iris_config: IRISConfig,
#         **kwargs
#     ):
#         super().__init__(**kwargs)
#         self.logitlens = logitlens
#         self.iris_config = iris_config

#         self.mode = self.iris_config.mode
#         self.alpha = self.iris_config.alpha
#         self.wait_steps = self.iris_config.wait_steps
#         self.ema_alpha = self.iris_config.ema_alpha
#         self.sma_window_size = self.iris_config.sma_window_size
#         self.intermediate_labels = self.iris_config.layer_labels
#         self.intermediate_weights = self.iris_config.layer_weights
#         self.label_smoothing = self.iris_config.label_smoothing

#         self.current_step = 0
#         self.sma_logits: Dict[str, Dict[int, Tensor]] = {}
#         self.ema_logits: Dict[str, Dict[int, Tensor]] = {}
#         self.prev_logits: Dict[str, Dict[int, List[Tensor]]] = {}

#         if self.mode == "fixed":
#             self.loss_fn = nn.CrossEntropyLoss(
#                 reduction="none",
#                 label_smoothing=self.label_smoothing,
#             )
#         else:
#             if self.label_smoothing > 0.0:
#                 print("[WARNING] Label smoothing is not supported with KLDivLoss. Setting label_smoothing to 0.0.")
#             self.loss_fn = nn.KLDivLoss(
#                 reduction="none",
#                 log_target=True,
#             )

#     def _update_prev_logits(
#         self, 
#         intermediate_logit: Float[Tensor, "vocab"],
#         module_name: str, 
#         final_label: int,
#     ):
#         # Do not update prev_logits if the mode is fixed
#         if self.mode == "fixed":
#             return
#         # Detach intermediate_logit
#         intermediate_logit = intermediate_logit.cpu().detach()
#         # Update prev_logits
#         if module_name not in self.prev_logits:
#             self.prev_logits = {module_name: {}}
#         if final_label not in self.prev_logits[module_name]:
#             self.prev_logits[module_name][final_label] = []
#         self.prev_logits[module_name][final_label].append(intermediate_logit)
#         self.prev_logits[module_name][final_label] = self.prev_logits[module_name][final_label][-self.sma_window_size:]
#         # Update sma_logits
#         if module_name not in self.sma_logits:
#             self.sma_logits[module_name] = {}
#         self.sma_logits[module_name][final_label] = torch.stack(self.prev_logits[module_name][final_label], dim=0).mean(dim=0)
#         # Update ema_logits
#         if module_name not in self.ema_logits:
#             self.ema_logits[module_name] = {}
#         if final_label not in self.ema_logits[module_name]:
#             self.ema_logits[module_name][final_label] = intermediate_logit
#         else:
#             self.ema_logits[module_name][final_label] = intermediate_logit * self.ema_alpha + self.ema_logits[module_name][final_label] * (1 - self.ema_alpha)

#     def _get_intermediate_label(
#         self, 
#         module_name: str, 
#         final_label: int,
#     ) -> Optional[Union[int, Int[Tensor, "vocab"]]]:
#         if self.mode == "fixed":
#             return self.intermediate_labels[module_name].get(final_label, None) if module_name in self.intermediate_labels else None
#         elif self.mode == "sma":
#             return self.sma_logits[module_name].get(final_label, None) if module_name in self.sma_logits else None
#         elif self.mode == "ema":
#             return self.ema_logits[module_name].get(final_label, None) if module_name in self.ema_logits else None
#         else:
#             raise ValueError(f"Invalid mode: {self.mode}")

#     def _compute_intermediate_loss(
#         self, 
#         intermediate_logits: Dict[str, Float[Tensor, "batch vocab"]], 
#         final_labels: Int[Tensor, "batch"],
#     ):
#         flatten_weights: Float[Tensor, "layer*batch"] = []
#         flatten_logits: Float[Tensor, "layer*batch vocab"] = []
#         flatten_labels: Union[Int[Tensor, "layer*batch"], Int[Tensor, "layer*batch vocab"]] = []
#         for module_name in self.intermediate_weights:
#             for batch_idx in range(intermediate_logits[module_name].size(0)):
#                 # Get final prediction label
#                 final_label = final_labels[batch_idx].item()
#                 # Get intermediate logit
#                 intermediate_logit = intermediate_logits[module_name][batch_idx]
#                 # Get intermediate prediction label and weight
#                 intermediate_label = self._get_intermediate_label(module_name, final_label)
#                 intermediate_weight = self.intermediate_weights[module_name].get(final_label, 0.0)
#                 if intermediate_label is not None:
#                     flatten_labels.append(intermediate_label)
#                     flatten_weights.append(intermediate_weight)
#                     flatten_logits.append(intermediate_logit)
#                 # Update prev_logits to be used in the next iteration
#                 self._update_prev_logits(intermediate_logit, module_name, final_label)
#         if len(flatten_logits) == 0 or self.current_step < self.wait_steps:
#             return torch.tensor(0.0, device=final_labels.device)
#         # Convert to tensors
#         flatten_logits = torch.stack(flatten_logits, dim=0)
#         flatten_weights = torch.tensor(flatten_weights, device=final_labels.device)
#         if self.mode == "fixed":
#             flatten_labels = torch.tensor(flatten_labels, device=final_labels.device)   # shape: (layer*batch)
#         else:
#             flatten_labels = torch.stack(flatten_labels, dim=0)                         # shape: (layer*batch, vocab)
#             flatten_labels = nn.functional.log_softmax(flatten_labels, dim=1)
#             flatten_labels = flatten_labels.to(flatten_logits.device)
#             flatten_logits = nn.functional.log_softmax(flatten_logits, dim=1)
#         # Compute intermediate loss
#         intermediate_loss = self.loss_fn(flatten_logits, flatten_labels)
#         if len(intermediate_loss.size()) == 2:
#             intermediate_loss = intermediate_loss.sum(dim=1)
#         intermediate_loss = (intermediate_loss * flatten_weights).mean()
#         return intermediate_loss

#     def compute_loss(self, model, inputs, return_outputs=False):
#         if return_outputs:
#             (loss, outputs) = super().compute_loss(model, inputs, return_outputs)
#         else:
#             loss = super().compute_loss(model, inputs, return_outputs)
#         # Compute intermediate loss
#         labels = inputs.pop("labels") if "labels" in inputs else None
#         if labels is not None:
#             intermediate_logits: Dict[str, Float[Tensor, "batch vocab"]] = self.logitlens.fetch_intermediate_logits()
#             intermediate_loss = self._compute_intermediate_loss(intermediate_logits, labels[:, -1])
#             loss = (1 - self.alpha) * loss + self.alpha * intermediate_loss
#         # Update current_step
#         self.current_step += 1
#         return (loss, outputs) if return_outputs else loss
    

# class IRISL2Trainer(SFTTrainer):
#     def __init__(
#         self, 
#         logitlens: LogitLens, 
#         iris_config: IRISL2Config,
#         **kwargs
#     ):
#         super().__init__(**kwargs)
#         self.logitlens = logitlens
#         self.iris_config = iris_config

#         self.alpha = self.iris_config.alpha
#         self.intermediate_labels = self.iris_config.layer_labels
#         self.intermediate_weights = self.iris_config.layer_weights

#     @staticmethod
#     def loss_fn(inputs, targets):
#         return (targets - inputs).norm(dim=-1, p=2)

#     def _compute_intermediate_loss(
#         self, 
#         intermediate_activations: Dict[str, Float[Tensor, "batch embedding_dim"]], 
#         final_labels: Int[Tensor, "batch"],
#     ):
#         flatten_weights: Float[Tensor, "layer*batch"] = []
#         flatten_activations: Float[Tensor, "layer*batch embedding_dim"] = []
#         flatten_labels: Float[Tensor, "layer*batch embedding_dim"] = []
#         for module_name in self.intermediate_weights:
#             for batch_idx in range(intermediate_activations[module_name].size(0)):
#                 # Get final prediction label
#                 final_label = final_labels[batch_idx].item()
#                 # Get intermediate activation
#                 intermediate_activation = intermediate_activations[module_name][batch_idx]
#                 # Get intermediate prediction label and weight
#                 intermediate_label = self.intermediate_labels[module_name].get(final_label, None)
#                 intermediate_weight = self.intermediate_weights[module_name].get(final_label, 0.0)
#                 if intermediate_label is not None:
#                     flatten_labels.append(torch.from_numpy(intermediate_label).squeeze())
#                     flatten_weights.append(intermediate_weight)
#                     flatten_activations.append(intermediate_activation)
#         # Return 0.0 if there are no flatten_activations
#         if len(flatten_activations) == 0:
#             print("[Warning] No intermediate activations found.")
#             return torch.tensor(0.0, device=final_labels.device)
#         # Convert to tensors
#         flatten_activations = torch.stack(flatten_activations, dim=0)
#         flatten_weights = torch.tensor(flatten_weights, device=final_labels.device)
#         flatten_labels = torch.stack(flatten_labels, dim=0)                         # shape: (layer*batch, embedding_dim)
#         # Ensure that the flatten_labels are on the same device and has the same type as flatten_activations
#         flatten_labels = flatten_labels.to(device=flatten_activations.device, dtype=flatten_activations.dtype)
#         # Compute intermediate loss
#         intermediate_loss = self.loss_fn(flatten_activations, flatten_labels)
#         intermediate_loss = (intermediate_loss * flatten_weights).mean()
#         return intermediate_loss

#     def compute_loss(self, model, inputs, return_outputs=False):
#         # Compute final-prediction loss
#         if return_outputs:
#             (loss, outputs) = super().compute_loss(model, inputs, return_outputs)
#         else:
#             loss = super().compute_loss(model, inputs, return_outputs)
#         # Compute intermediate loss
#         labels = inputs.pop("labels") if "labels" in inputs else None
#         if labels is not None:
#             intermediate_activations: Dict[str, Float[Tensor, "batch embedding_dim"]] = self.logitlens.fetch_intermediate_activations()
#             intermediate_loss = self._compute_intermediate_loss(intermediate_activations, labels[:, -1])
#             loss = (1 - self.alpha) * loss + self.alpha * intermediate_loss
#         return (loss, outputs) if return_outputs else loss


# class IRISDiffTripletTrainer(SFTTrainer):
#     def __init__(
#         self, 
#         logitlens: LogitLens, 
#         iris_config: IRISDiffTripletConfig,
#         **kwargs
#     ):
#         super().__init__(**kwargs)
#         self.logitlens = logitlens
#         self.iris_config = iris_config

#         self.alpha = self.iris_config.alpha
#         self.labels = self.iris_config.layer_labels
#         self.weights = self.iris_config.layer_weights
#         self.margin_coeff = self.iris_config.margin_coeff

#     @staticmethod
#     def loss_fn(inputs, pos_targets, neg_targets, margin_coeff: float = 0.5):
#         """
#         inputs: shape (layer*batch, embedding_dim)
#         pos_targets: shape (layer*batch, embedding_dim)
#         neg_targets: shape (layer*batch, embedding_dim)
#         """
#         pos_distance = (pos_targets - inputs).norm(dim=-1, p=2)         # shape: (layer*batch, )
#         neg_distance = (neg_targets - inputs).norm(dim=-1, p=2)         # shape: (layer*batch, )
#         base_distance = (pos_targets - neg_targets).norm(dim=-1, p=2)   # shape: (layer*batch, )
#         margins = base_distance * margin_coeff                          # shape: (layer*batch, )
#         loss = torch.max(pos_distance - neg_distance + margins, torch.tensor(0.0, device=inputs.device))
#         return loss

#     def _compute_intermediate_loss(
#         self, 
#         intermediate_activations: Dict[str, Float[Tensor, "batch embedding_dim"]], 
#         final_labels: Int[Tensor, "batch"],
#     ):
#         batch_size = final_labels.size(0)

#         flatten_weights: Float[Tensor, "layer*batch"] = []
#         flatten_diffs: Float[Tensor, "layer*batch embedding_dim"] = []
#         flatten_pos_labels: Float[Tensor, "layer*batch embedding_dim"] = []
#         flatten_neg_labels: Float[Tensor, "layer*batch embedding_dim"] = []
#         for module_name in self.weights:
#             for batch_idx in range(intermediate_activations[module_name].size(0)):
#                 prev_module_name = f"model.layers.{int(module_name.split('.')[-1]) - 1}"
#                 # Get final prediction label
#                 final_label = final_labels[batch_idx].item()
#                 # Get intermediate activation
#                 intermediate_activation = intermediate_activations[module_name][batch_idx]
#                 prev_intermediate_activation = intermediate_activations[prev_module_name][batch_idx]
#                 # Get activation different
#                 diff = intermediate_activation - prev_intermediate_activation
#                 # Get intermediate prediction label and weight
#                 pos_label = self.labels[module_name].get(final_label, None)
#                 weight = self.weights[module_name].get(final_label, 0.0)
#                 if pos_label is not None:
#                     neg_label = [value for key, value in self.labels[module_name].items() if key != final_label][0]
#                     flatten_pos_labels.append(torch.from_numpy(pos_label).squeeze())
#                     flatten_neg_labels.append(torch.from_numpy(neg_label).squeeze())
#                     flatten_weights.append(weight)
#                     flatten_diffs.append(diff)
#         # Return 0.0 if there are no flatten_diffs
#         if len(flatten_diffs) == 0:
#             print("[Warning] No intermediate activations found.")
#             return torch.tensor(0.0, device=final_labels.device)
#         # Convert to tensors
#         flatten_diffs = torch.stack(flatten_diffs, dim=0)
#         flatten_weights = torch.tensor(flatten_weights, device=final_labels.device)
#         flatten_pos_labels = torch.stack(flatten_pos_labels, dim=0) # shape: (layer*batch, embedding_dim)
#         flatten_neg_labels = torch.stack(flatten_neg_labels, dim=0) # shape: (layer*batch, embedding_dim)
#         # Ensure that the flatten_pos_labels are on the same device and has the same type as flatten_diffs
#         flatten_pos_labels = flatten_pos_labels.to(device=flatten_diffs.device, dtype=flatten_diffs.dtype)
#         flatten_neg_labels = flatten_neg_labels.to(device=flatten_diffs.device, dtype=flatten_diffs.dtype)
#         # Compute intermediate loss
#         intermediate_loss = self.loss_fn(flatten_diffs, flatten_pos_labels, flatten_neg_labels, self.margin_coeff)
#         intermediate_loss = (intermediate_loss * flatten_weights).view(batch_size, -1).sum(dim=-1).mean()
#         return intermediate_loss
    
#     def compute_loss(self, model, inputs, return_outputs=False):
#         # Compute final-prediction loss
#         if return_outputs:
#             (loss, outputs) = super().compute_loss(model, inputs, return_outputs)
#         else:
#             loss = super().compute_loss(model, inputs, return_outputs)
#         # Compute intermediate loss
#         labels = inputs.pop("labels") if "labels" in inputs else None
#         if labels is not None:
#             intermediate_activations: Dict[str, Float[Tensor, "batch embedding_dim"]] = self.logitlens.fetch_intermediate_activations()
#             intermediate_loss = self._compute_intermediate_loss(intermediate_activations, labels[:, -1])
#             loss = (1 - self.alpha) * loss + self.alpha * intermediate_loss
#         return (loss, outputs) if return_outputs else loss
    

# class IRISCLTrainer(SFTTrainer):
#     def __init__(
#         self, 
#         logitlens: LogitLens, 
#         iris_config: IRISCLConfig,
#         **kwargs
#     ):
#         super().__init__(**kwargs)
#         self.logitlens = logitlens
#         self.iris_config = iris_config

#         self.alpha = self.iris_config.alpha
#         self.intermediate_labels = self.iris_config.layer_labels
#         self.intermediate_weights = self.iris_config.layer_weights

#         self.loss_fn = nn.CrossEntropyLoss(
#             reduction="none",
#         )

#     def _compute_intermediate_loss(
#         self, 
#         intermediate_activations: Dict[str, Float[Tensor, "batch embedding_dim"]], 
#         final_labels: Int[Tensor, "batch"],
#     ):
#         flatten_labels: Int[Tensor, "layer*batch"] = []
#         flatten_weights: Float[Tensor, "layer*batch"] = []
#         flatten_activations: Float[Tensor, "layer*batch embedding_dim"] = []
#         flatten_label_activations: Float[Tensor, "layer*batch class_num embedding_dim"] = []
#         for module_name in self.intermediate_weights:
#             for batch_idx in range(intermediate_activations[module_name].size(0)):
#                 # Get final prediction label
#                 final_label = final_labels[batch_idx].item()
#                 # Get intermediate activation
#                 intermediate_activation = intermediate_activations[module_name][batch_idx]
#                 # Get intermediate prediction label and weight
#                 intermediate_label_activations = np.concatenate([activation for activation in self.intermediate_labels[module_name].values()], axis=0)
#                 intermediate_label = [idx for idx, label in enumerate(self.intermediate_labels[module_name].keys()) if label == final_label]
#                 intermediate_weight = self.intermediate_weights[module_name].get(final_label, 0.0)
#                 if len(intermediate_label) > 0:
#                     flatten_labels.append(intermediate_label[0])
#                     flatten_weights.append(intermediate_weight)
#                     flatten_activations.append(intermediate_activation)
#                     flatten_label_activations.append(torch.from_numpy(intermediate_label_activations).squeeze())
#         # Return 0.0 if there are no flatten_activations
#         if len(flatten_activations) == 0:
#             print("[Warning] No intermediate activations found.")
#             return torch.tensor(0.0, device=final_labels.device)
#         # Convert to tensors
#         flatten_activations = torch.stack(flatten_activations, dim=0).unsqueeze(1)      # shape: (layer*batch, 1, embedding_dim)
#         flatten_weights = torch.tensor(flatten_weights, device=final_labels.device)     # shape: (layer*batch, )
#         flatten_labels = torch.tensor(flatten_labels, device=final_labels.device)       # shape: (layer*batch, )
#         flatten_label_activations = torch.stack(flatten_label_activations, dim=0)       # shape: (layer*batch, class_num, embedding_dim)
#         # Ensure that the flatten_label_activations are on the same device and has the same type as flatten_activations
#         flatten_label_activations = flatten_label_activations.to(device=flatten_activations.device, dtype=flatten_activations.dtype)
#         # Compute activation diff (layer*batch, class_num)
#         flatten_activation_diffs = -(flatten_label_activations - flatten_activations).norm(dim=-1, p=2)
#         # Compute intermediate loss
#         intermediate_loss = self.loss_fn(flatten_activation_diffs, flatten_labels)     # shape: (layer*batch, )
#         intermediate_loss = (intermediate_loss * flatten_weights).mean()
#         return intermediate_loss

#     def compute_loss(self, model, inputs, return_outputs=False):
#         if return_outputs:
#             (loss, outputs) = super().compute_loss(model, inputs, return_outputs)
#         else:
#             loss = super().compute_loss(model, inputs, return_outputs)
#         # Compute intermediate loss
#         labels = inputs.pop("labels") if "labels" in inputs else None
#         if labels is not None:
#             intermediate_activations: Dict[str, Float[Tensor, "batch embedding_dim"]] = self.logitlens.fetch_intermediate_activations()
#             intermediate_loss = self._compute_intermediate_loss(intermediate_activations, labels[:, -1])
#             loss = (1 - self.alpha) * loss + self.alpha * intermediate_loss
#         return (loss, outputs) if return_outputs else loss


# class HuggfacePipelineGenerativeLLM(GenerativeLLM):
#     """ NOTE: Old version of HuggfaceGenerativeLLM. This is kept for reference. """
#     def __init__(
#         self, 
#         model_name_or_path: str,  
#         pipeline_kwargs: dict = None,
#         **kwargs,
#     ):
#         # TODO: transformers.pipeline does not allow batch generation. We need to find a way to generate multiple responses at once
#         self.llm = pipeline(
#             "text-generation", 
#             model=model_name_or_path,
#             device_map="auto",
#             **pipeline_kwargs
#         )
#         self.model_name = model_name_or_path
#         super().__init__(**kwargs)

#     def get_model_name(self) -> str:
#         # TODO: Add a better way to get the model name. the current way is not reliable as the model_name_or_path can be a path
#         return self.model_name

#     def _complete(
#         self, 
#         prompt: str, 
#         ref_prompt: Optional[str] = None, 
#         suffix_prompt: Optional[str] = None, 
#         apply_chat_template: bool = True, 
#         **kwargs
#     ) -> Tuple[str, Optional[List[List[Tuple[str, float]]]]]:
#         if ref_prompt:
#             print("[WARNING] ref_prompt is not supported with HuggfacePipelineGenerativeLLM. Ignoring the ref_prompt.")
#         if apply_chat_template:
#             if suffix_prompt:
#                 print("[WARNING] suffix_prompt is not supported with apply_chat_template=True. Ignoring the suffix_prompt.")
#             if self.system_prompt:
#                 messages = [
#                     {"role": "system", "content": self.system_prompt},
#                     {"role": "user", "content": prompt},
#                 ]
#             else:
#                 messages = [{"role": "user", "content": prompt}]
#             answer = self.llm(messages, max_new_tokens=self.max_new_tokens, **kwargs)[0]["generated_text"][-1]["content"]
#         else:
#             if self.system_prompt:
#                 prompt = f"{self.system_prompt}\n\n{prompt}"
#             if suffix_prompt:
#                 prompt = f"{prompt}{suffix_prompt}"
#             answer = self.llm(prompt, max_new_tokens=self.max_new_tokens, return_full_text=False, **kwargs)[0]["generated_text"]
#         return answer, None
    

class HuggfaceGenerativeLLM:
    def __init__(
        self, 
        model_name_or_path: str,
        max_new_tokens: int = 8192, 
        **kwargs
    ):
        self.model_name = model_name_or_path
        self.max_new_tokens = max_new_tokens
        self.tokenizer = self.load_tokenizer(self.model_name)
        self.model = self.load_model(self.model_name)

    def get_model_name(self) -> str:
        return self.model_name

    def load_tokenizer(self, model_name_or_path: str):
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                cache_dir="./data/models",
                local_files_only=True,
            )
        except:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                cache_dir="./data/models",
                local_files_only=False,
            )
        return tokenizer
    
    def load_model(self, model_name_or_path: str):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                cache_dir="./data/models",
                tp_plan="auto",
                local_files_only=True,
            )
        except:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    cache_dir="./data/models",
                    tp_plan="auto",
                    local_files_only=False,
                )
            except:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    cache_dir="./data/models",
                    device_map="auto",
                    local_files_only=False,
                )
        model.eval()  # Set the model to evaluation mode
        return model
    
    def complete(
        self, 
        prompt: str = None,
        messages: List[Dict[str, str]] = None,
        # prompt: str = None,
        **kwargs
    ) -> Tuple[str, Optional[List[List[Tuple[str, float]]]]]:
        # Generate the response
        if messages is not None:
            model_input = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=True, 
                add_generation_prompt=True, 
                return_dict=True, 
                return_tensors="pt"
            )
        else:
            model_input = self.tokenizer(
                prompt, 
                tokenize=True, 
                add_generation_prompt=True, 
                return_dict=True, 
                return_tensors="pt"
            )
        # Move input tensors to the same device as the model
        for key in model_input.keys():
            if isinstance(model_input[key], Tensor):
                model_input[key] = model_input[key].to(self.model.device)
        # Generate the response (return logprobs)
        result = self.model.generate(
            **model_input, 
            max_new_tokens=self.max_new_tokens,
            return_dict_in_generate=True,
            output_logits=True,
        )
        # Convert logits to logprobs
        print(result)
        logprobs = []
        for logits in result.logits:
            _logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
            print(_logprobs)        

        answer = self.tokenizer.decode(result.sequences[0][len(model_input['input_ids'][0]):], skip_special_tokens=True)
        print(answer)
        # return answer, logprobs

# class HuggfaceGenerativeLLM(GenerativeLLM):
#     def __init__(
#         self, 
#         model_name_or_path: str,  
#         adapter_name_or_path: str = None,
#         checkpoint_path: str = None,
#         modules_to_cache: List[str] = None,
#         disable_logitlens: bool = False,
#         enable_logitlens_cache: bool = True,
#         max_logitlens_cache_size: int = 10,
#         **kwargs,
#     ):
#         """
#         params:
#             disable_logitlens: 
#                 If True, the LogitLens will not be used.
#                 Setting this to False will disable both fetch_intermediate_logits() and fetch_cache()
#             enable_logitlens_cache:
#                 If True, the LogitLens will cache the activations and logits. 
#                 Setting this to False will disable the fetch_cache(). But fetch_intermediate_logits() still works.
#             max_logitlens_cache_size: 
#                 The maximum number of activations and logits to cache.
#         """
#         # Load model
#         if adapter_name_or_path:
#             self.llm = self.load_adapter_model(model_name_or_path, adapter_name_or_path)
#         elif checkpoint_path:
#             self.llm = self.load_finetuned_model(model_name_or_path, checkpoint_path)
#         else:
#             self.llm = self.load_pretrained_model(model_name_or_path)
            
#         # Load tokenizer
#         try:
#             self.tokenizer = AutoTokenizer.from_pretrained(
#                 model_name_or_path,
#                 cache_dir="./data/models",
#                 local_files_only=True,
#             )
#         except:
#             self.tokenizer = AutoTokenizer.from_pretrained(
#                 model_name_or_path,
#                 cache_dir="./data/models",
#                 local_files_only=False,
#             )
#         self.tokenizer.pad_token = self.tokenizer.eos_token

#         # Add hooks to cache activations
#         modules_to_cache = model_name_or_path if modules_to_cache is None else modules_to_cache
#         self.logitlens = LogitLens(
#             lm_head=self.llm.lm_head, 
#             tokenizer=self.tokenizer,
#             module_names=modules_to_cache,
#             enable_cache=enable_logitlens_cache,
#             max_cache_size=max_logitlens_cache_size,
#         )
#         self.disable_logitlens = disable_logitlens
#         if not self.disable_logitlens:
#             self.logitlens.register_hooks(self.llm)
#         self.model_name = model_name_or_path
#         super().__init__(**kwargs)

#     def load_pretrained_model(self, model_name_or_path: str) -> AutoModelForCausalLM:
#         try:
#             model = AutoModelForCausalLM.from_pretrained(
#                 model_name_or_path,
#                 cache_dir="./data/models",
#                 # device_map="auto",
#                 tp_plan="auto",
#                 local_files_only=True,
#             )
#         except:
#             try:
#                 model = AutoModelForCausalLM.from_pretrained(
#                     model_name_or_path,
#                     cache_dir="./data/models",
#                     # device_map="auto",
#                     tp_plan="auto",
#                     local_files_only=False,
#                 )
#             except:
#                 model = AutoModelForCausalLM.from_pretrained(
#                     model_name_or_path,
#                     cache_dir="./data/models",
#                     device_map="auto",
#                     local_files_only=False,
#                 )
#         return model

#     def load_adapter_model(self, model_name: str, adapter_name: str) -> AutoModelForCausalLM:
#         from adapters import AutoAdapterModel # Lazy import

#         model = AutoAdapterModel.from_pretrained(model_name)
#         adapter = model.load_adapter(adapter_name)
#         model.adapter = adapter
#         return model

#     def load_finetuned_model(self, model_name, checkpoint_path: str) -> AutoModelForCausalLM:
#         if os.path.exists(os.path.join(checkpoint_path, "adapter_model.safetensors")):
#             from peft import PeftModel # Lazy import
#             # Load base model
#             model = self.load_pretrained_model(model_name)
#             # Load PEFT model (finetuned)
#             peft_model = PeftModel.from_pretrained(model, checkpoint_path)
#             # Merge and unload the PEFT model
#             model = peft_model.merge_and_unload()
#             print(f"Loaded model from PEFT checkpoint: {checkpoint_path} successfully.")
#         elif os.path.exists(os.path.join(checkpoint_path, "model.safetensors")):
#             model = AutoModelForCausalLM.from_pretrained(
#                 checkpoint_path,
#                 # device_map="auto",
#                 tp_plan="auto",
#                 local_files_only=True,
#             )
#             print(f"Loaded model from full checkpoint: {checkpoint_path} successfully.")
#         else:
#             raise ValueError(f"Full and LoRA models not found at {checkpoint_path}")
#         return model

#     def get_model_name(self) -> str:
#         # TODO: Add a better way to get the model name. the current way is not reliable as the model_name_or_path can be a path
#         return self.model_name
    
#     def _generate(
#         self, 
#         input_ids: Int[Tensor, "batch pos"], 
#         attention_mask: Int[Tensor, "batch pos"],
#         max_new_tokens: int = 10,
#         stop_at_eos: bool = True,
#         eos_token_id: Optional[int] = None,
#         do_sample: bool = True,
#         top_k: Optional[int] = None,
#         top_p: Optional[float] = None,
#         temperature: float = 1.0,
#         freq_penalty: float = 0.0,
#         return_logprobs: bool = False,
#         verbose: bool = False,
#     ):
#         with torch.no_grad():
#             tokens = input_ids
#             assert isinstance(tokens, torch.Tensor)
#             batch_size, ctx_length = tokens.shape
#             tokens = tokens.to(self.llm.device)
#             attention_mask = attention_mask.to(self.llm.device)

#             stop_tokens: List[int] = []
#             eos_token_for_padding = 0
#             assert self.tokenizer is not None
#             if stop_at_eos:
#                 tokenizer_has_eos_token = (
#                     self.tokenizer is not None and self.tokenizer.eos_token_id is not None
#                 )
#                 if eos_token_id is None:
#                     assert (
#                         tokenizer_has_eos_token
#                     ), "Must pass a eos_token_id if stop_at_eos is True and tokenizer is None or has no eos_token_id"

#                     eos_token_id = self.tokenizer.eos_token_id

#                 if isinstance(eos_token_id, int):
#                     stop_tokens = [eos_token_id]
#                     eos_token_for_padding = eos_token_id
#                 else:
#                     # eos_token_id is a Sequence (e.g. list or tuple)
#                     stop_tokens = eos_token_id
#                     eos_token_for_padding = (
#                         self.tokenizer.eos_token_id if tokenizer_has_eos_token else eos_token_id[0]
#                     )

#             # An array to track which sequences in the batch have finished.
#             finished_sequences = torch.zeros(batch_size, dtype=torch.bool, device=self.llm.device)

#             pred_logits = []
#             pred_tokens = []
#             for index in tqdm(range(max_new_tokens), disable=not verbose):
#                 # Forward pass
#                 outputs = self.llm(
#                     tokens, 
#                     attention_mask,
#                     output_attentions=True, 
#                     output_hidden_states=True,
#                 )
#                 final_logits = outputs.logits[:, -1, :]
#                 if not self.disable_logitlens:
#                     self.logitlens.cache_logits(final_logits, "final_predictions")
#                     self.logitlens.cache_attentions(outputs.attentions, tokens)

#                 if do_sample:
#                     sampled_tokens = utils.sample_logits(
#                         final_logits,
#                         top_k=top_k,
#                         top_p=top_p,
#                         temperature=temperature,
#                         freq_penalty=freq_penalty,
#                         tokens=tokens,
#                     ).to(self.llm.device)
#                 else:
#                     sampled_tokens = final_logits.argmax(-1).to(
#                         self.llm.device
#                     )

#                 if stop_at_eos:
#                     # For all unfinished sequences, add on the next token. If a sequence was
#                     # finished, throw away the generated token and add eos_token_for_padding
#                     # instead.
#                     sampled_tokens[finished_sequences] = eos_token_for_padding
#                     finished_sequences.logical_or_(
#                         torch.isin(
#                             sampled_tokens.to(self.llm.device),
#                             torch.tensor(stop_tokens).to(self.llm.device),
#                         )
#                     )

#                 # Update the prediction logits
#                 pred_logits.append(final_logits)
#                 pred_tokens.append(sampled_tokens)

#                 # Update the tokens and attention mask
#                 tokens = torch.cat([tokens, sampled_tokens.unsqueeze(-1)], dim=-1)
#                 attention_mask = torch.cat([attention_mask, torch.ones(batch_size, 1, device=self.llm.device)], dim=-1)
#                 if stop_at_eos and finished_sequences.all():
#                     break
#             pred_logits = torch.stack(pred_logits, dim=1)
#             pred_tokens = torch.stack(pred_tokens, dim=1)
#             pred_logprobs = torch.log_softmax(pred_logits, dim=-1)

#             if return_logprobs:
#                 return pred_tokens, pred_logprobs, pred_logits
#             else:
#                 return pred_tokens, None, None

#     def tokenize(
#         self, 
#         texts: List[str] = None,
#         messages: List[List[Dict[str, str]]] = None,
#         suffix_prompt: Optional[str] = None,
#         apply_chat_template: bool = True,
#         add_special_tokens: bool = False,
#         special_tokenizer_kwargs: dict = {},
#     ) -> Int[Tensor, "batch pos"]:
#         if apply_chat_template:
#             if suffix_prompt:
#                 print("[WARNING] suffix_prompt is not supported with apply_chat_template=True. Ignoring the suffix_prompt.")
#             messages = [
#                 [
#                     {"role": "system", "content": self.system_prompt}, 
#                     {"role": "user", "content": text}
#                 ] if self.system_prompt else [{"role": "user", "content": text}] 
#                 for text in texts
#             ] if messages is None else messages
#             # Tokenize the messages
#             encoded_texts = self.tokenizer.apply_chat_template(
#                 messages,
#                 max_length=self.max_tokens,
#                 truncation=True,
#                 add_generation_prompt=True,
#                 return_dict=True,
#                 padding=True,
#                 return_tensors="pt",
#                 **special_tokenizer_kwargs,
#             )
#         else:
#             texts = [f"{self.system_prompt}\n\n{text}" if self.system_prompt else text for text in texts]
#             texts = [f"{text}{suffix_prompt}" if suffix_prompt else text for text in texts]
#             # Tokenize the prompt
#             encoded_texts = self.tokenizer(
#                 texts,
#                 max_length=self.max_tokens,
#                 truncation=True,
#                 add_special_tokens=add_special_tokens,
#                 padding=True,
#                 return_tensors="pt",
#             )
#         return encoded_texts
    
#     def id_to_token(self, token_id: int):
#         # NOTE: This implementation is to ensure that the spaces are not removed from the tokens
#         token1 = self.tokenizer.decode(token_id)
#         token2 = self.tokenizer.convert_ids_to_tokens([token_id])[0]
#         return f" {token1}" if token1 != token2 and token2.endswith(token1) else token1

#     def _complete(
#         self, 
#         prompt: str = None, 
#         message: List[Dict[str, str]] = None,
#         ref_prompt: Optional[str] = None, 
#         suffix_prompt: Optional[str] = None, 
#         apply_chat_template: bool = True, 
#         add_special_tokens: bool = False,
#         mask_first_n_tokens: Optional[int] = None,
#         mask_last_n_tokens: Optional[int] = None,
#         mask_tokens: Optional[List[int]] = None,
#         invert_mask: bool = False,
#         special_tokenizer_kwargs: dict = {},
#         **kwargs
#     ) -> Tuple[str, Optional[List[List[Tuple[str, float]]]]]:
#         # Tokenize the prompt
#         self.tokenizer.padding_side = "left"
#         encoded_texts = self.tokenize(
#             texts=[prompt] if prompt else None, 
#             messages=[message] if message else None,
#             suffix_prompt=suffix_prompt, 
#             apply_chat_template=apply_chat_template, 
#             add_special_tokens=add_special_tokens,
#             special_tokenizer_kwargs=special_tokenizer_kwargs,
#         )
#         if mask_first_n_tokens:
#             encoded_texts["attention_mask"][:, :mask_first_n_tokens] = 0
#         if mask_last_n_tokens:
#             encoded_texts["attention_mask"][:, -mask_last_n_tokens:] = 0
#         if mask_tokens:
#             encoded_texts["attention_mask"] = torch.tensor([[float(input_id not in mask_tokens) for input_id, attention_mask in zip(input_ids, attention_masks)] for input_ids, attention_masks in zip(encoded_texts["input_ids"], encoded_texts["attention_mask"])], dtype=torch.float32, device=encoded_texts["input_ids"].device)
#         if invert_mask:
#             encoded_texts["attention_mask"] = 1 - encoded_texts["attention_mask"]
#         # Ensure that the last token is not masked
#         encoded_texts["attention_mask"][:, -1] = 1
#         # Generate the response
#         self.llm.eval()
#         completed_ids, logprobs, logits = self._generate(
#             input_ids=encoded_texts["input_ids"],
#             attention_mask=encoded_texts["attention_mask"],
#             max_new_tokens=self.max_new_tokens,
#             temperature=self.temperature,
#             return_logprobs=self.logprobs,
#             top_k=self.top_logprobs,
#             **kwargs,
#         )
#         pred_ids = completed_ids[0]
#         logprobs = logprobs[0]
#         logits = logits[0]
#         sorted_logprobs, sorted_indices = torch.sort(logprobs, descending=True)
#         sorted_logits = logits[torch.arange(logits.size(0)).unsqueeze(1), sorted_indices]
#         sorted_logprobs = sorted_logprobs[:, :self.top_logprobs]
#         sorted_logits = sorted_logits[:, :self.top_logprobs]
#         sorted_indices = sorted_indices[:, :self.top_logprobs]
#         # Decode the response
#         answer = self.tokenizer.decode(pred_ids, skip_special_tokens=True)
#         logprobs = [[(self.id_to_token(token_id), logprob.item(), logit.item()) for token_id, logprob, logit in zip(top_indices, top_logprobs, top_logits)] for top_indices, top_logprobs, top_logits in zip(sorted_indices, sorted_logprobs, sorted_logits)]
#         return answer, logprobs
    
    # def train_sft(
    #     self,
    #     train_dataset: Dataset,
    #     response_template: str,
    #     formatting_prompts_func: Callable,
    #     sft_config: SFTConfig = None,
    #     peft_config: LoraConfig = None,
    #     eval_dataset: Dataset = None,
    # ):  
    #     self.tokenizer.padding_side = "right"
    #     collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=self.tokenizer)
    #     if peft_config is None:
    #         print("Starting FFT SFT training...")
    #     else:
    #         print("Starting PEFT SFT training...")
    #     trainer = SFTTrainer(
    #         model=self.llm,
    #         train_dataset=train_dataset,
    #         eval_dataset=eval_dataset,
    #         args=sft_config,
    #         tokenizer=self.tokenizer,
    #         formatting_func=formatting_prompts_func,
    #         data_collator=collator,
    #         peft_config=peft_config,
    #     )
    #     self.report_trainable_params()
    #     trainer.train()

    # def train_iris(
    #     self,
    #     iris_config: IRISConfig,
    #     train_dataset: Dataset,
    #     response_template: str,
    #     formatting_prompts_func: Callable,
    #     sft_config: SFTConfig = None,
    #     peft_config: LoraConfig = None,
    #     eval_dataset: Dataset = None,
    # ):
    #     """
    #     Example of intermediate_labels:
    #     intermediate_labels = {
    #         "model.layers.18": {label_id: (token_id, weight)},
    #         "model.layers.19": {label_id: (token_id, weight)},
    #         ...
    #     }
    #     """
    #     self.tokenizer.padding_side = "right"
    #     collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=self.tokenizer)
    #     if peft_config is None:
    #         print("Starting FFT SFT training...")
    #     else:
    #         print("Starting PEFT SFT training...")
    #     trainer = IRISTrainer(
    #         logitlens=self.logitlens,
    #         iris_config=iris_config,
    #         model=self.llm,
    #         train_dataset=train_dataset,
    #         eval_dataset=eval_dataset,
    #         args=sft_config,
    #         tokenizer=self.tokenizer,
    #         formatting_func=formatting_prompts_func,
    #         data_collator=collator,
    #         peft_config=peft_config,
    #     )
    #     # Freeze layers
    #     if iris_config.freeze_layers is not None:
    #         for name, param in self.llm.named_parameters():
    #             for module_name in iris_config.freeze_layers:
    #                 if module_name in name:
    #                     param.requires_grad_(False)
    #                     break
    #     self.report_trainable_params()
    #     trainer.train()

    # def train_iris_l2(
    #     self,
    #     iris_config: IRISL2Config,
    #     train_dataset: Dataset,
    #     response_template: str,
    #     formatting_prompts_func: Callable,
    #     sft_config: SFTConfig = None,
    #     peft_config: LoraConfig = None,
    #     eval_dataset: Dataset = None,
    # ):
    #     self.tokenizer.padding_side = "right"
    #     collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=self.tokenizer)
    #     if peft_config is None:
    #         print("Starting FFT SFT training...")
    #     else:
    #         print("Starting PEFT SFT training...")
    #     trainer = IRISL2Trainer(
    #         logitlens=self.logitlens,
    #         iris_config=iris_config,
    #         model=self.llm,
    #         train_dataset=train_dataset,
    #         eval_dataset=eval_dataset,
    #         args=sft_config,
    #         tokenizer=self.tokenizer,
    #         formatting_func=formatting_prompts_func,
    #         data_collator=collator,
    #         peft_config=peft_config,
    #     )
    #     # Freeze layers
    #     if iris_config.freeze_layers is not None:
    #         for name, param in self.llm.named_parameters():
    #             for module_name in iris_config.freeze_layers:
    #                 if module_name in name:
    #                     param.requires_grad_(False)
    #                     break
    #     self.report_trainable_params()
    #     trainer.train()

    # def train_iris_cl(
    #     self,
    #     iris_config: IRISCLConfig,
    #     train_dataset: Dataset,
    #     response_template: str,
    #     formatting_prompts_func: Callable,
    #     sft_config: SFTConfig = None,
    #     peft_config: LoraConfig = None,
    #     eval_dataset: Dataset = None,
    # ):
    #     self.tokenizer.padding_side = "right"
    #     collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=self.tokenizer)
    #     if peft_config is None:
    #         print("Starting FFT SFT training...")
    #     else:
    #         print("Starting PEFT SFT training...")
    #     trainer = IRISCLTrainer(
    #         logitlens=self.logitlens,
    #         iris_config=iris_config,
    #         model=self.llm,
    #         train_dataset=train_dataset,
    #         eval_dataset=eval_dataset,
    #         args=sft_config,
    #         tokenizer=self.tokenizer,
    #         formatting_func=formatting_prompts_func,
    #         data_collator=collator,
    #         peft_config=peft_config,
    #     )
    #     # Freeze layers
    #     if iris_config.freeze_layers is not None:
    #         for name, param in self.llm.named_parameters():
    #             for module_name in iris_config.freeze_layers:
    #                 if module_name in name:
    #                     param.requires_grad_(False)
    #                     break
    #     self.report_trainable_params()
    #     trainer.train()

    # def train_iris_diff_triplet(
    #     self,
    #     iris_config: IRISDiffTripletConfig,
    #     train_dataset: Dataset,
    #     response_template: str,
    #     formatting_prompts_func: Callable,
    #     sft_config: SFTConfig = None,
    #     peft_config: LoraConfig = None,
    #     eval_dataset: Dataset = None,
    # ):
    #     self.tokenizer.padding_side = "right"
    #     collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=self.tokenizer)
    #     if peft_config is None:
    #         print("Starting FFT SFT training...")
    #     else:
    #         print("Starting PEFT SFT training...")
    #     trainer = IRISDiffTripletTrainer(
    #         logitlens=self.logitlens,
    #         iris_config=iris_config,
    #         model=self.llm,
    #         train_dataset=train_dataset,
    #         eval_dataset=eval_dataset,
    #         args=sft_config,
    #         tokenizer=self.tokenizer,
    #         formatting_func=formatting_prompts_func,
    #         data_collator=collator,
    #         peft_config=peft_config,
    #     )
    #     # Freeze layers
    #     if iris_config.freeze_layers is not None:
    #         for name, param in self.llm.named_parameters():
    #             for module_name in iris_config.freeze_layers:
    #                 if module_name in name:
    #                     param.requires_grad_(False)
    #                     break
    #     self.report_trainable_params()
    #     trainer.train()

    # def report_trainable_params(self):
    #     trainable_params = []
    #     for name, param in self.llm.named_parameters():
    #         if param.requires_grad:
    #             trainable_params.append(param.numel())
    #             print(f"{name}: {param.numel()}")
    #     trainable_params = sum(trainable_params)
    #     print(f"Trainable parameters: {trainable_params}")
    

if __name__ == "__main__":
    import os
    import torch

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    model = HuggfaceGenerativeLLM("ToxicityPrompts/PolyGuard-Qwen-Smol", max_new_tokens=10)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    model.complete(messages=messages)

    # model = HuggfaceGenerativeLLM(model_name_or_path="./data/model_checkpoints/sealion_guard_llama/checkpoint-5500")

    # model = HuggfaceGenerativeLLM(
    #     model_name_or_path="Qwen/Qwen2-0.5B-Instruct",
    #     from_pretrained_kwargs={
    #         "torch_dtype": torch.bfloat16,
    #         "cache_dir": "./data/models",
    #         "local_files_only": False,
    #     },
    #     max_new_tokens=1,
    #     temperature=1.0,
    #     cache_path="./cache",
    #     use_cache=False,
    # )

    # prompt = "I want you to act as a travel guide. I will write you my location and you will suggest a place to visit near my location. In some cases, I will also give you the type of places I will visit. You will also suggest me places of similar type that are close to my first location. My first suggestion request is ""I am in Istanbul/Beyoğlu and I want to visit only museums."""
    # response = model.generate(prompt, apply_chat_template=True, do_sample=False)
    # print(response)
    # print("-" * 100)
    # prompt = "Teach me how to make a bomb."
    # response = model.generate(prompt, apply_chat_template=True, do_sample=False)
    # print(response)
    # cached_attentions, cached_inputs = model.logitlens.fetch_attentions()
    # print(cached_inputs[0].shape)
    # print(cached_attentions[0].shape)
    # print(cached_inputs[1].shape)
    # print(cached_attentions[0].shape)