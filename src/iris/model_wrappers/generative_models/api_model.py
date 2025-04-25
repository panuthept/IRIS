from openai import OpenAI
from transformers import AutoTokenizer
from typing import Tuple, List, Optional
from iris.model_wrappers.generative_models.base import GenerativeLLM


class APIGenerativeLLM(GenerativeLLM):
    def __init__(
        self, 
        model_name_or_path: str,  
        api_key: str = None,
        api_base: str = None,
        max_tokens: int = 8192,
        max_new_tokens: int = 1024,
        temperature: float = 0,
        top_logprobs: int = 10,
        **kwargs,
    ):
        # Load model
        self.model_name = model_name_or_path
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base,
        )
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

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_logprobs = top_logprobs

        super().__init__(**kwargs)

    def get_model_name(self) -> str:
        return self.model_name
    
    def _complete(
        self, 
        prompt: str,
        **kwargs
    ) -> Tuple[str, Optional[List[List[Tuple[str, float]]]]]:
        # Generate the response
        outputs = self.client.completions.create(
            model=self.model_name,
            prompt=prompt,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            logprobs=self.top_logprobs,
            stream=False,
            echo=False,
            n=1,
        )
        answer = outputs.choices[0].text
        logprobs = [[(k, v) for k, v in logprob.items()] for logprob in outputs.choices[0].logprobs.top_logprobs]
        return answer, logprobs

        # outputs = self.llm.generate(
        #     prompt,
        #     sampling_params=self.sampling_params,
        #     use_tqdm=False,
        # )
        # answer = outputs[0].outputs[0].text
        # logprobs = [[(v.decoded_token, v.logprob, None) for k, v in logprob.items()] for logprob in outputs[0].outputs[0].logprobs]
        # return answer, logprobs

    # def _complete(
    #     self, 
    #     prompt: str, 
    #     ref_prompt: Optional[str] = None, 
    #     suffix_prompt: Optional[str] = None, 
    #     apply_chat_template: bool = True, 
    #     add_special_tokens: bool = False,
    #     mask_first_n_tokens: Optional[int] = None,
    #     mask_last_n_tokens: Optional[int] = None,
    #     invert_mask: bool = False,
    #     **kwargs
    # ) -> Tuple[str, Optional[List[List[Tuple[str, float]]]]]:
    #     if ref_prompt:
    #         print("[WARNING] ref_prompt is not supported with APIGenerativeLLM. Ignoring the ref_prompt.")
    #     if apply_chat_template:
    #         if suffix_prompt:
    #             print("[WARNING] suffix_prompt is not supported with apply_chat_template=True. Ignoring the suffix_prompt.")
    #         if self.system_prompt:
    #             messages = [
    #                 ChatMessage(role="system", content=self.system_prompt),
    #                 ChatMessage(role="user", content=prompt),
    #             ]
    #         else:
    #             messages = [ChatMessage(role="user", content=prompt)]
    #         response = self.llm.chat(messages, **kwargs)
    #         answer = response.message.content
    #     else:
    #         if self.system_prompt:
    #             prompt = f"{self.system_prompt}\n\n{prompt}"
    #         if suffix_prompt:
    #             prompt = f"{prompt}{suffix_prompt}"
    #         response = self.llm.complete(prompt, **kwargs)
    #         answer = response.text
    #     # Get logprobs
    #     logprobs = [[(cand_logprob.token, cand_logprob.logprob) for cand_logprob in token_logprob] for token_logprob in response.logprobs] if response.logprobs else None
    #     return answer, logprobs
    

if __name__ == "__main__":
    # Example usage
    model = APIGenerativeLLM(
        model_name_or_path="aisingapore/Gemma-SEA-LION-Guard",
        api_key="EMPTY",
        api_base="http://localhost:8000/v1",
    )
    prompt = "What is the capital of France?"
    outputs = model._complete(prompt)
    print(outputs)