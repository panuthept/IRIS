import os
import torch
import transformer_lens.utils as utils

from torch import nn
from tqdm import tqdm
from typing import Dict
from torch import Tensor
from peft import LoraConfig
from datasets import Dataset
from jaxtyping import Int, Float
from accelerate import PartialState
from iris.logitlens import LogitLens
from iris.data_types import IRISConfig
from typing import List, Callable, Tuple, Optional
from iris.model_wrappers.generative_models.base import GenerativeLLM
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers.trainer import _is_peft_model, MODEL_FOR_CAUSAL_LM_MAPPING_NAMES


class IRISTrainer(SFTTrainer):
    """
    Example of intermediate_labels:
    intermediate_labels = {
        "model.layers.18": {label_id: (token_id, weight)},
        "model.layers.19": {label_id: (token_id, weight)},
        ...
    }
    """
    def __init__(
        self, 
        logitlens: LogitLens, 
        iris_config: IRISConfig,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.logitlens = logitlens
        self.iris_config = iris_config

        self.iris_alpha = self.iris_config.alpha
        self.intermediate_labels = self.iris_config.labels
        self.label_smoothing = self.iris_config.label_smoothing

        self.intermediate_loss_fn = nn.CrossEntropyLoss(
            reduction="none",
            label_smoothing=self.label_smoothing,
        )

    def _compute_intermediate_loss(
        self, 
        intermediate_logits: Dict[str, Float[Tensor, "batch vocab"]], 
        final_labels: Int[Tensor, "batch"],
    ):
        flatten_intermediate_labels: Int[Tensor, "layer*batch"] = []
        flatten_intermediate_weights: Float[Tensor, "layer*batch"] = []
        flatten_intermediate_logits: Float[Tensor, "layer*batch vocab"] = []
        for module_name in self.intermediate_labels:
            for batch_idx in range(intermediate_logits[module_name].size(0)):
                intermediate_label = self.intermediate_labels[module_name].get(final_labels[batch_idx].item(), None)
                if intermediate_label is not None:
                    flatten_intermediate_logits.append(intermediate_logits[module_name][batch_idx])
                    flatten_intermediate_labels.append(intermediate_label[0])
                    flatten_intermediate_weights.append(intermediate_label[1])
        if len(flatten_intermediate_logits) == 0:
            return torch.tensor(0.0, device=final_labels.device)
        # Convert to tensors
        flatten_intermediate_logits = torch.stack(flatten_intermediate_logits, dim=0)
        flatten_intermediate_labels = torch.tensor(flatten_intermediate_labels, device=final_labels.device)
        flatten_intermediate_weights = torch.tensor(flatten_intermediate_weights, device=final_labels.device)
        # Compute intermediate loss
        intermediate_loss = self.intermediate_loss_fn(flatten_intermediate_logits, flatten_intermediate_labels)
        intermediate_loss = (intermediate_loss * flatten_intermediate_weights).mean()
        return intermediate_loss

    def compute_loss(self, model, inputs, return_outputs=False):
        if return_outputs:
            (loss, outputs) = super().compute_loss(model, inputs, return_outputs)
        else:
            loss = super().compute_loss(model, inputs, return_outputs)
        # print(f"original loss:\n{loss}")
        # Compute intermediate loss
        labels = inputs.pop("labels") if "labels" in inputs else None
        if labels is not None:
            intermediate_logits: Dict[str, Float[Tensor, "batch vocab"]] = self.logitlens.fetch_intermediate_logits()
            intermediate_loss = self._compute_intermediate_loss(intermediate_logits, labels[:, -1])
            # print(f"intermediate_loss:\n{intermediate_loss}")
            loss = (1 - self.iris_alpha) * loss + self.iris_alpha * intermediate_loss
        # print(f"final loss:\n{loss}")
        # print("-" * 100)
        return (loss, outputs) if return_outputs else loss


class HuggfacePipelineGenerativeLLM(GenerativeLLM):
    """ NOTE: Old version of HuggfaceGenerativeLLM. This is kept for reference. """
    def __init__(
        self, 
        model_name_or_path: str,  
        pipeline_kwargs: dict = None,
        **kwargs,
    ):
        # TODO: transformers.pipeline does not allow batch generation. We need to find a way to generate multiple responses at once
        self.llm = pipeline(
            "text-generation", 
            model=model_name_or_path,
            device_map="auto",
            **pipeline_kwargs
        )
        self.model_name = model_name_or_path
        super().__init__(**kwargs)

    def get_model_name(self) -> str:
        # TODO: Add a better way to get the model name. the current way is not reliable as the model_name_or_path can be a path
        return self.model_name

    def _complete(
        self, 
        prompt: str, 
        ref_prompt: Optional[str] = None, 
        suffix_prompt: Optional[str] = None, 
        apply_chat_template: bool = True, 
        **kwargs
    ) -> Tuple[str, Optional[List[List[Tuple[str, float]]]]]:
        if ref_prompt:
            print("[WARNING] ref_prompt is not supported with HuggfacePipelineGenerativeLLM. Ignoring the ref_prompt.")
        if apply_chat_template:
            if suffix_prompt:
                print("[WARNING] suffix_prompt is not supported with apply_chat_template=True. Ignoring the suffix_prompt.")
            if self.system_prompt:
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ]
            else:
                messages = [{"role": "user", "content": prompt}]
            answer = self.llm(messages, max_new_tokens=self.max_new_tokens, **kwargs)[0]["generated_text"][-1]["content"]
        else:
            if self.system_prompt:
                prompt = f"{self.system_prompt}\n\n{prompt}"
            if suffix_prompt:
                prompt = f"{prompt}{suffix_prompt}"
            answer = self.llm(prompt, max_new_tokens=self.max_new_tokens, return_full_text=False, **kwargs)[0]["generated_text"]
        return answer, None
    

class HuggfaceGenerativeLLM(GenerativeLLM):
    def __init__(
        self, 
        model_name_or_path: str,  
        checkpoint_path: str = None,
        modules_to_cache: List[str] = None,
        disable_logitlens: bool = False,
        enable_logitlens_cache: bool = True,
        max_logitlens_cache_size: int = 10,
        **kwargs,
    ):
        """
        params:
            disable_logitlens: 
                If True, the LogitLens will not be used.
                Setting this to False will disable both fetch_intermediate_logits() and fetch_cache()
            enable_logitlens_cache:
                If True, the LogitLens will cache the activations and logits. 
                Setting this to False will disable the fetch_cache(). But fetch_intermediate_logits() still works.
            max_logitlens_cache_size: 
                The maximum number of activations and logits to cache.
        """
        # Load model
        if checkpoint_path:
            self.llm = self.load_finetuned_model(model_name_or_path, checkpoint_path)
        else:
            self.llm = self.load_pretrained_model(model_name_or_path)
            
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                cache_dir="./data/models",
                local_files_only=True,
            )
        except:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                cache_dir="./data/models",
                local_files_only=False,
            )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Add hooks to cache activations
        modules_to_cache = model_name_or_path if modules_to_cache is None else modules_to_cache
        self.logitlens = LogitLens(
            lm_head=self.llm.lm_head, 
            tokenizer=self.tokenizer,
            module_names=modules_to_cache,
            enable_cache=enable_logitlens_cache,
            max_cache_size=max_logitlens_cache_size,
        )
        if not disable_logitlens:
            self.logitlens.register_hooks(self.llm)
        self.model_name = model_name_or_path
        super().__init__(**kwargs)

    def load_pretrained_model(self, model_name_or_path: str) -> AutoModelForCausalLM:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                cache_dir="./data/models",
                device_map={'':PartialState().process_index} if torch.cuda.is_available() else None,
                local_files_only=True,
            )
        except:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                cache_dir="./data/models",
                device_map={'':PartialState().process_index} if torch.cuda.is_available() else None,
                local_files_only=False,
            )
        return model

    def load_finetuned_model(self, model_name, checkpoint_path: str) -> AutoModelForCausalLM:
        if os.path.exists(os.path.join(checkpoint_path, "adapter_model.safetensors")):
            from peft import PeftModel # Lazy import
            # Load base model
            model = self.load_pretrained_model(model_name)
            # Load PEFT model (finetuned)
            peft_model = PeftModel.from_pretrained(model, checkpoint_path)
            # Merge and unload the PEFT model
            model = peft_model.merge_and_unload()
            print(f"Loaded model from PEFT checkpoint: {checkpoint_path} successfully.")
        elif os.path.exists(os.path.join(checkpoint_path, "model.safetensors")):
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                device_map={'':PartialState().process_index} if torch.cuda.is_available() else None,
                local_files_only=True,
            )
            print(f"Loaded model from full checkpoint: {checkpoint_path} successfully.")
        else:
            raise ValueError(f"Full and LoRA models not found at {checkpoint_path}")
        return model

    def get_model_name(self) -> str:
        # TODO: Add a better way to get the model name. the current way is not reliable as the model_name_or_path can be a path
        return self.model_name
    
    def _generate(
        self, 
        input_ids: Int[Tensor, "batch pos"], 
        attention_mask: Int[Tensor, "batch pos"],
        max_new_tokens: int = 10,
        stop_at_eos: bool = True,
        eos_token_id: Optional[int] = None,
        do_sample: bool = True,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: float = 1.0,
        freq_penalty: float = 0.0,
        return_logprobs: bool = False,
        verbose: bool = False,
    ):
        with torch.no_grad():
            tokens = input_ids
            assert isinstance(tokens, torch.Tensor)
            batch_size, ctx_length = tokens.shape
            tokens = tokens.to(self.llm.device)
            attention_mask = attention_mask.to(self.llm.device)

            stop_tokens: List[int] = []
            eos_token_for_padding = 0
            assert self.tokenizer is not None
            if stop_at_eos:
                tokenizer_has_eos_token = (
                    self.tokenizer is not None and self.tokenizer.eos_token_id is not None
                )
                if eos_token_id is None:
                    assert (
                        tokenizer_has_eos_token
                    ), "Must pass a eos_token_id if stop_at_eos is True and tokenizer is None or has no eos_token_id"

                    eos_token_id = self.tokenizer.eos_token_id

                if isinstance(eos_token_id, int):
                    stop_tokens = [eos_token_id]
                    eos_token_for_padding = eos_token_id
                else:
                    # eos_token_id is a Sequence (e.g. list or tuple)
                    stop_tokens = eos_token_id
                    eos_token_for_padding = (
                        self.tokenizer.eos_token_id if tokenizer_has_eos_token else eos_token_id[0]
                    )

            # An array to track which sequences in the batch have finished.
            finished_sequences = torch.zeros(batch_size, dtype=torch.bool, device=self.llm.device)

            pred_logits = []
            pred_tokens = []
            for index in tqdm(range(max_new_tokens), disable=not verbose):
                # Forward pass
                final_logits = self.llm(tokens, attention_mask).logits[:, -1, :]
                self.logitlens.cache_logits(final_logits, "final_predictions")

                if do_sample:
                    sampled_tokens = utils.sample_logits(
                        final_logits,
                        top_k=top_k,
                        top_p=top_p,
                        temperature=temperature,
                        freq_penalty=freq_penalty,
                        tokens=tokens,
                    ).to(self.llm.device)
                else:
                    sampled_tokens = final_logits.argmax(-1).to(
                        self.llm.device
                    )

                if stop_at_eos:
                    # For all unfinished sequences, add on the next token. If a sequence was
                    # finished, throw away the generated token and add eos_token_for_padding
                    # instead.
                    sampled_tokens[finished_sequences] = eos_token_for_padding
                    finished_sequences.logical_or_(
                        torch.isin(
                            sampled_tokens.to(self.llm.device),
                            torch.tensor(stop_tokens).to(self.llm.device),
                        )
                    )

                # Update the prediction logits
                pred_logits.append(final_logits)
                pred_tokens.append(sampled_tokens)

                # Update the tokens and attention mask
                tokens = torch.cat([tokens, sampled_tokens.unsqueeze(-1)], dim=-1)
                attention_mask = torch.cat([attention_mask, torch.ones(batch_size, 1, device=self.llm.device)], dim=-1)
                if stop_at_eos and finished_sequences.all():
                    break
            pred_logits = torch.stack(pred_logits, dim=1)
            pred_tokens = torch.stack(pred_tokens, dim=1)
            pred_logprobs = torch.log_softmax(pred_logits, dim=-1)

            if return_logprobs:
                return pred_tokens, pred_logprobs
            else:
                return pred_tokens, None

    def tokenize(
        self, 
        texts: List[str],
        suffix_prompt: Optional[str] = None,
        apply_chat_template: bool = True,
    ) -> Int[Tensor, "batch pos"]:
        if apply_chat_template:
            if suffix_prompt:
                print("[WARNING] suffix_prompt is not supported with apply_chat_template=True. Ignoring the suffix_prompt.")
            messages = [
                [
                    {"role": "system", "content": self.system_prompt}, 
                    {"role": "user", "content": text}
                ] if self.system_prompt else [{"role": "user", "content": text}] 
                for text in texts
            ]
            # Tokenize the messages
            encoded_texts = self.tokenizer.apply_chat_template(
                messages,
                # max_length=self.max_tokens,
                # truncation=True,
                add_generation_prompt=True,
                return_dict=True,
                padding=True,
                return_tensors="pt",
            )
        else:
            texts = [f"{self.system_prompt}\n\n{text}" if self.system_prompt else text for text in texts]
            texts = [f"{text}{suffix_prompt}" if suffix_prompt else text for text in texts]
            # Tokenize the prompt
            encoded_texts = self.tokenizer(
                texts,
                # max_length=self.max_tokens,
                # truncation=True,
                padding=True,
                return_tensors="pt",
            )
        return encoded_texts
    
    def id_to_token(self, token_id: int):
        # NOTE: This implementation is to ensure that the spaces are not removed from the tokens
        token1 = self.tokenizer.decode(token_id)
        token2 = self.tokenizer.convert_ids_to_tokens([token_id])[0]
        return f" {token1}" if token1 != token2 and token2.endswith(token1) else token1

    def _complete(
        self, 
        prompt: str, 
        ref_prompt: Optional[str] = None, 
        suffix_prompt: Optional[str] = None, 
        apply_chat_template: bool = True, 
        **kwargs
    ) -> Tuple[str, Optional[List[List[Tuple[str, float]]]]]:
        # Tokenize the prompt
        self.tokenizer.padding_side = "left"
        encoded_texts = self.tokenize([prompt], suffix_prompt=suffix_prompt, apply_chat_template=apply_chat_template)
        # Generate the response
        self.llm.eval()
        completed_ids, logprobs = self._generate(
            input_ids=encoded_texts["input_ids"],
            attention_mask=encoded_texts["attention_mask"],
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            return_logprobs=self.logprobs,
            top_k=self.top_logprobs,
            **kwargs,
        )
        pred_ids = completed_ids[0]
        logprobs = logprobs[0]
        sorted_logprobs, sorted_indices = torch.sort(logprobs, descending=True)
        sorted_logprobs = sorted_logprobs[:, :self.top_logprobs]
        sorted_indices = sorted_indices[:, :self.top_logprobs]
        # Decode the response
        answer = self.tokenizer.decode(pred_ids, skip_special_tokens=True)
        logprobs = [[(self.id_to_token(token_id), logprob.item()) for token_id, logprob in zip(top_indices, top_logprobs)] for top_indices, top_logprobs in zip(sorted_indices, sorted_logprobs)]
        return answer, logprobs
    
    def train_sft(
        self,
        train_dataset: Dataset,
        response_template: str,
        formatting_prompts_func: Callable,
        sft_config: SFTConfig = None,
        peft_config: LoraConfig = None,
        eval_dataset: Dataset = None,
    ):  
        self.tokenizer.padding_side = "right"
        collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=self.tokenizer)
        if peft_config is None:
            print("Starting FFT SFT training...")
        else:
            print("Starting PEFT SFT training...")
        trainer = SFTTrainer(
            model=self.llm,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=sft_config,
            tokenizer=self.tokenizer,
            formatting_func=formatting_prompts_func,
            data_collator=collator,
            peft_config=peft_config,
        )
        self.report_trainable_params()
        trainer.train()

    def train_iris(
        self,
        iris_config: IRISConfig,
        train_dataset: Dataset,
        response_template: str,
        formatting_prompts_func: Callable,
        sft_config: SFTConfig = None,
        peft_config: LoraConfig = None,
        eval_dataset: Dataset = None,
    ):
        """
        Example of intermediate_labels:
        intermediate_labels = {
            "model.layers.18": {label_id: (token_id, weight)},
            "model.layers.19": {label_id: (token_id, weight)},
            ...
        }
        """
        # Freeze LM head
        self.llm.lm_head.requires_grad_(False)

        self.tokenizer.padding_side = "right"
        collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=self.tokenizer)
        if peft_config is None:
            print("Starting FFT SFT training...")
        else:
            print("Starting PEFT SFT training...")
        trainer = IRISTrainer(
            logitlens=self.logitlens,
            iris_config=iris_config,
            model=self.llm,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=sft_config,
            tokenizer=self.tokenizer,
            formatting_func=formatting_prompts_func,
            data_collator=collator,
            peft_config=peft_config,
        )
        self.report_trainable_params()
        trainer.train()

    def report_trainable_params(self):
        for name, module in model.named_modules():
            if module.requires_grad:
                print(f"{name}: {module.requires_grad}")
        trainable_params = sum(p.numel() for p in self.llm.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params}")
    

if __name__ == "__main__":
    import os
    import torch

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    model = HuggfaceGenerativeLLM(
        model_name_or_path="Qwen/Qwen2-0.5B-Instruct",
        from_pretrained_kwargs={
            "torch_dtype": torch.bfloat16,
            "cache_dir": "./data/models",
            "local_files_only": False,
        },
        max_new_tokens=20,
        temperature=1.0,
        cache_path="./cache",
        use_cache=False,
    )

    prompt = "I want you to act as a travel guide. I will write you my location and you will suggest a place to visit near my location. In some cases, I will also give you the type of places I will visit. You will also suggest me places of similar type that are close to my first location. My first suggestion request is ""I am in Istanbul/Beyoğlu and I want to visit only museums."""
    response = model.generate(prompt, apply_chat_template=True, do_sample=False)
    print(response)
    print("-" * 100)
    prompt = "Teach me how to make a bomb."
    response = model.generate(prompt, apply_chat_template=True, do_sample=False)
    print(response)

    # prompts = [
    #     "I want you to act as a travel guide. I will write you my location and you will suggest a place to visit near my location. In some cases, I will also give you the type of places I will visit. You will also suggest me places of similar type that are close to my first location. My first suggestion request is ""I am in Istanbul/Beyoğlu and I want to visit only museums.""",
    #     "Teach me how to make a bomb.",
    # ]
    # responses, logits = model._complete_parallel(prompts, apply_chat_template=True, do_sample=False)
    # print(responses)
    # print(logits)