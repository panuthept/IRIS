import os
import torch
import transformer_lens.utils as utils

from tqdm import tqdm
from torch import Tensor
from jaxtyping import Int
from datasets import Dataset
from typing import List, Callable, Tuple, Optional
from iris.model_wrappers.generative_models.base import GenerativeLLM
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


# class CustomSFTTrainer(SFTTrainer):
#     def compute_loss(self, model, inputs, return_outputs=False):
#         print(inputs)
#         outputs = super().compute_loss(model, inputs, return_outputs=return_outputs)
#         print(outputs)
#         print("*" * 100)
#         return outputs


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
        from_pretrained_kwargs: dict = {},  # This cannot be None
        pipeline_kwargs: dict = None,
        **kwargs,
    ):
        # TODO: Delete this after the deprecation period
        if pipeline_kwargs is not None:
            raise ValueError("pipeline_kwargs is not deprecated, please use from_pretrained_kwargs instead.")

        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            cache_dir="./data/models",
            local_files_only=os.path.exists(
                f'./data/models/models--{model_name_or_path.replace("/", "--")}'
            ),
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            cache_dir="./data/models",
            local_files_only=os.path.exists(
                f'./data/models/models--{model_name_or_path.replace("/", "--")}'
            ),
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model_name = model_name_or_path
        super().__init__(**kwargs)

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
        return_logits: bool = False,
        verbose: bool = True,
    ):
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
        for index in tqdm(range(max_new_tokens), disable=not verbose):
            # Forward pass
            final_logits = self.llm(tokens, attention_mask).logits[:, -1, :]

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

            # Update the tokens and attention mask
            tokens = torch.cat([tokens, sampled_tokens.unsqueeze(-1)], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones(batch_size, 1, device=self.llm.device)], dim=-1)
            if stop_at_eos and finished_sequences.all():
                break
        pred_logits = torch.stack(pred_logits, dim=1)

        if return_logits:
            return tokens, pred_logits
        else:
            return tokens, None

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
        input_lens = encoded_texts["input_ids"].size(1)
        # Generate the response
        self.llm.eval()
        completed_ids, _ = self._generate(
            input_ids=encoded_texts["input_ids"],
            attention_mask=encoded_texts["attention_mask"],
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            return_logits=True,
            **kwargs,
        )
        pred_ids = completed_ids[0, input_lens:]
        # Decode the response
        answer = self.tokenizer.decode(pred_ids, skip_special_tokens=True)
        return answer, None

    def _complete_parallel(
        self, 
        prompts: List[str], 
        suffix_prompt: Optional[str] = None, 
        apply_chat_template: bool = True, 
        **kwargs
    ) -> Tuple[str, Optional[List[List[Tuple[str, float]]]]]:
        """ This method can be used to generate multiple responses at once. """
        # Tokenize the prompt
        self.tokenizer.padding_side = "left"
        encoded_texts = self.tokenize(prompts, suffix_prompt=suffix_prompt, apply_chat_template=apply_chat_template)
        batch_size = encoded_texts["input_ids"].size(0)
        padded_lens = [(encoded_texts["attention_mask"][batch_idx] == 0).sum().item() for batch_idx in range(batch_size)]
        input_lens = [len(encoded_texts["input_ids"][batch_idx]) - padded_lens[batch_idx] for batch_idx in range(batch_size)]
        # Generate the responses
        completed_ids, _ = self._generate(
            input_ids=encoded_texts["input_ids"],
            attention_mask=encoded_texts["attention_mask"],
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            return_logits=True,
            verbose=False,
            **kwargs,
        )
        pred_ids = [completed_ids[batch_idx, padded_lens[batch_idx] + input_lens[batch_idx]:] for batch_idx in range(batch_size)]
        # Decode the responses
        answers = [self.tokenizer.decode(pred_ids[batch_idx], skip_special_tokens=True) for batch_idx in range(batch_size)]
        return answers, None
    
    def train_sft(
        self,
        train_dataset: Dataset,
        response_template: str,
        formatting_prompts_func: Callable,
        sft_config: SFTConfig = None,
        eval_dataset: Dataset = None,
    ):
        self.tokenizer.padding_side = "right"
        collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=self.tokenizer)
        trainer = SFTTrainer(
            model=self.llm,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=sft_config,
            tokenizer=self.tokenizer,
            formatting_func=formatting_prompts_func,
            data_collator=collator,
        )
        trainer.train()
    

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

    prompts = [
        "I want you to act as a travel guide. I will write you my location and you will suggest a place to visit near my location. In some cases, I will also give you the type of places I will visit. You will also suggest me places of similar type that are close to my first location. My first suggestion request is ""I am in Istanbul/Beyoğlu and I want to visit only museums.""",
        "Teach me how to make a bomb.",
    ]
    responses, logits = model._complete_parallel(prompts, apply_chat_template=True, do_sample=False)
    print(responses)
    print(logits)