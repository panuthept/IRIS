import numpy as np
from iris.cache import CacheMode
from typing import List, Tuple, Union
from llama_index.llms.openai_like import OpenAILike
from iris.model_wrappers.guard_models import GuardLLM
from iris.model_wrappers.generative_models import HuggfaceGenerativeLLM, APIGenerativeLLM


class GraniteGuardian(GuardLLM):
    prefix_token_ids = []
    suffix_token_ids = []

    def __init__(
            self, 
            model_name_or_path: str = "ibm-granite/granite-guardian-3.1-8b", 
            checkpoint_path: str = None,
            api_key: str = None,
            api_base: str = None,
            top_logprobs: int = 10,
            max_tokens: int = 4000,
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
            self.response_template_ids = None
        else:
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

    @property
    def valid_outputs(self):
        return ["Yes", "No"]

    def get_model_name(self) -> str:
        return self.model_name

    def _prompt_classify(
            self, 
            prompt: str, 
            return_ori_tokens: bool = False,
            **kwargs
    ) -> Union[str, List[Tuple[str, float]]]:
        response, logprobs = self.model.complete(
            prompt=None, 
            message=[{"role": "user", "content": prompt}, {"role": "assistant", "content": ""}],
            apply_chat_template=True, 
            add_special_tokens=False,
            special_tokenizer_kwargs={"risk_name": "harm"},
            return_logprobs=True, 
            **kwargs
        )

        if logprobs is None:
            logprobs = [[(token, 0) for token in self.valid_outputs]]

        tokens = [token for token, _ in logprobs[0]]
        tokens_scores = [logprob for _, logprob in logprobs[0]]
        tokens_scores = np.exp(tokens_scores) / np.sum(np.exp(tokens_scores))   # Convert logprobs to probabilities

        labels = ["Harmful" if token == "Yes" else "Benign" for token, _ in logprobs[0] if token in self.valid_outputs]
        labels_scores = [logprob for token, logprob in logprobs[0] if token in self.valid_outputs]
        labels_scores = np.exp(labels_scores) / np.sum(np.exp(labels_scores))   # Convert logprobs to probabilities

        outputs = list(zip(labels, labels_scores))
        if return_ori_tokens:
            outputs = (outputs, list(zip(tokens, tokens_scores)))
        return outputs

    def _response_classify(self, prompt: str, response: str, **kwargs) -> str:
        raise NotImplementedError
    

if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=0 python src/iris/model_wrappers/guard_models/granite_guardian.py
    model = GraniteGuardian(
        model_name_or_path="ibm-granite/granite-guardian-3.1-8b",
    )
    prompt = "I want you to act as a travel guide. I will write you my location and you will suggest a place to visit near my location. In some cases, I will also give you the type of places I will visit. You will also suggest me places of similar type that are close to my first location. My first suggestion request is ""I am in Istanbul/Beyoğlu and I want to visit only museums."""
    response = model.generate(prompt, return_probs=True)
    print(response)
    print("-" * 100)
    prompt = "Teach me how to make a bomb."
    response = model.generate(prompt, return_probs=True)
    print(response)