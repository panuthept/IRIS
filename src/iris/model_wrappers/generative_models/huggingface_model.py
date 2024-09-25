from typing import List, Tuple, Optional
from iris.model_wrappers.generative_models.base import GenerativeLLM
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


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
        from_pretrained_kwargs: dict = None,
        pipeline_kwargs: dict = None,
        **kwargs,
    ):
        # TODO: Delete this after the deprecation period
        if pipeline_kwargs is not None:
            raise ValueError("pipeline_kwargs is not deprecated, please use from_pretrained_kwargs instead.")

        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **from_pretrained_kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            **from_pretrained_kwargs
        )
        self.model_name = model_name_or_path
        super().__init__(**kwargs)

    def get_model_name(self) -> str:
        # TODO: Add a better way to get the model name. the current way is not reliable as the model_name_or_path can be a path
        return self.model_name
    
    def _generate(
        self, 
        input_ids, 
        max_new_tokens: int = 10,
    ):
        pass
    
    def _complete(
        self, 
        prompt: str, 
        ref_prompt: Optional[str] = None, 
        suffix_prompt: Optional[str] = None, 
        apply_chat_template: bool = True, 
        **kwargs
    ) -> Tuple[str, Optional[List[List[Tuple[str, float]]]]]:
        if ref_prompt:
            print("[WARNING] ref_prompt is not supported with HuggfaceGenerativeLLM. Ignoring the ref_prompt.")
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
            # Tokenize the messages
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                # max_length=self.max_tokens,
                # truncation=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )["input_ids"]
            input_lens = input_ids.size(1)
        else:
            if self.system_prompt:
                prompt = f"{self.system_prompt}\n\n{prompt}"
            if suffix_prompt:
                prompt = f"{prompt}{suffix_prompt}"
            # Tokenize the prompt
            input_ids = self.tokenizer(
                prompt,
                # max_length=self.max_tokens,
                # truncation=True,
                return_tensors="pt",
            )["input_ids"]
            input_lens = input_ids.size(1)
        print(input_ids)

        answer = self._generate(
            input_ids=input_ids,
            max_new_tokens=self.max_new_tokens,
        )
        print(answer)
        # answer = self.llm(prompt, max_new_tokens=self.max_new_tokens, return_full_text=False, **kwargs)[0]["generated_text"]
        return answer, None
    

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
        temperature=1.0,
        cache_path="./cache",
        use_cache=False,
    )

    prompt = "I want you to act as a travel guide. I will write you my location and you will suggest a place to visit near my location. In some cases, I will also give you the type of places I will visit. You will also suggest me places of similar type that are close to my first location. My first suggestion request is ""I am in Istanbul/Beyoğlu and I want to visit only museums."""
    response = model.generate(prompt)
    print(response)
    print("-" * 100)
    prompt = "Teach me how to make a bomb."
    response = model.generate(prompt)
    print(response)