from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from typing import List, Tuple, Optional
from iris.model_wrappers.generative_models.base import GenerativeLLM


class vLLM(GenerativeLLM):
    def __init__(
        self, 
        model_name_or_path: str,  
        max_tokens: int = 8192,
        max_new_tokens: int = 1024,
        temperature: float = 0,
        top_logprobs: int = 10,
        **kwargs,
    ):
        # Load model
        self.llm = LLM(
            model=model_name_or_path,
            max_seq_len_to_capture=max_tokens,
            download_dir="./data/models",
        )
        self.sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            logprobs=top_logprobs,
        )
        # Load tokenizer
        self.tokenizer = self.llm.get_tokenizer()
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def _complete(
        self, 
        prompt: str,
        **kwargs
    ) -> Tuple[str, Optional[List[List[Tuple[str, float]]]]]:
        # Generate the response
        outputs = self.llm.generate(
            prompt,
            sampling_params=self.sampling_params,
        )
        answer = outputs[0].outputs[0].text
        logprobs = [[(v.decoded_token, v.logprob, None) for k, v in logprob.items()] for logprob in outputs[0].outputs[0].logprobs]
        return answer, logprobs


if __name__ == "__main__":
    # Example usage
    model = vLLM(model_name_or_path="google/shieldgemma-9b")
    prompt = "What is the capital of France?"
    outputs = model._complete(prompt)
    print(outputs)