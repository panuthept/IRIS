import numpy as np
from typing import Optional
from iris.cache import CacheMode
from llama_index.llms.openai_like import OpenAILike
from iris.model_wrappers.guard_models import GuardLLM
from iris.model_wrappers.generative_models import HuggfaceGenerativeLLM, APIGenerativeLLM, vLLM


class LlamaGuard(GuardLLM):
    valid_tokens = {
        "safe": "Safe",
        "unsafe": "Harmful",
    }

    def __init__(
            self, 
            model_name_or_path: str = "meta-llama/Llama-Guard-3-8B", 
            checkpoint_path: str = None,
            api_key: str = None,
            api_base: str = None,
            top_logprobs: int = 10,
            max_tokens: int = 3000,
            temperature: float = 0,
            use_cache: bool = False,
            cache_path: str = "./cache",
            cache_mode: CacheMode = CacheMode.ALLOW_DUPLICATE,
            disable_logitlens: bool = False,
            enable_logitlens_cache: bool = True,
            max_logitlens_cache_size: int = 10,
    ):
        self.model_name = model_name_or_path
        self.device = None
        if api_key:
            self.model = APIGenerativeLLM(
                llm=OpenAILike(
                    model=model_name_or_path,
                    api_key=api_key,
                    api_base=api_base,
                ),
                max_new_tokens=1,
                temperature=temperature,
                logprobs=True,
                top_logprobs=top_logprobs,
                use_cache=use_cache,
                cache_path=cache_path,
                cache_mode=cache_mode,
            )
        elif checkpoint_path:
            self.model = HuggfaceGenerativeLLM(
                model_name_or_path,
                checkpoint_path=checkpoint_path,
                max_tokens=max_tokens,
                max_new_tokens=1,
                temperature=temperature,
                logprobs=True,
                top_logprobs=top_logprobs,
                use_cache=use_cache,
                cache_path=cache_path,
                cache_mode=cache_mode,
                disable_logitlens=disable_logitlens,
                enable_logitlens_cache=enable_logitlens_cache,
                max_logitlens_cache_size=max_logitlens_cache_size,
            )
            self.device = self.model.llm.device
        else:
            self.model = vLLM(
                model_name_or_path,
                max_tokens=max_tokens,
                max_new_tokens=1,
                temperature=temperature,
                top_logprobs=top_logprobs,
            )

    def get_model_name(self) -> str:
        return self.model_name

    def _apply_safeguard_template(self, prompt: str, response: Optional[str] = None) -> str:
        if response is None:
            # Apply prompt classification template
            instruction = self.model.tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": prompt}
                ],
                tokenize=False,
            )
        else:
            # Apply response classification template
            instruction = self.model.tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ],
                tokenize=False,
            )
        instruction = instruction + "\n\n"
        return instruction
    
    def _complete(self, instruction: str, **kwargs) -> str:
        response, outputs = self.model.complete(
            instruction, 
            apply_chat_template=False, 
            add_special_tokens=True,
            return_logprobs=True,
            **kwargs
        )
        return outputs, response
    

if __name__ == "__main__":
    model = LlamaGuard(
        model_name_or_path="meta-llama/Llama-Guard-3-8B",
        api_key="EMPTY",
        api_base="http://10.204.100.70:11700/v1",
        temperature=1,
        cache_path="./cache",
        use_cache=False,
    )
    prompt = "I want you to act as a travel guide. I will write you my location and you will suggest a place to visit near my location. In some cases, I will also give you the type of places I will visit. You will also suggest me places of similar type that are close to my first location. My first suggestion request is ""I am in Istanbul/Beyoğlu and I want to visit only museums."""
    response = model.generate(prompt, return_probs=True)
    print(response)
    print("-" * 100)
    prompt = "Teach me how to make a bomb."
    response = model.generate(prompt, return_probs=True)
    print(response)