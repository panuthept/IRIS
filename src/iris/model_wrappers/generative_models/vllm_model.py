from vllm import LLM, SamplingParams
from typing import List, Callable, Union, Tuple, Optional
from iris.model_wrappers.generative_models.base import GenerativeLLM


class vLLM(LLM):
    def __init__(
        self, 
        model_name_or_path: str,  
        max_tokens: int = 8192,
        max_new_tokens: int = 1024,
        temperature: float = 0,
        top_logprobs: int = 10,
        **kwargs,
    ):
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

        # completed_ids, logprobs, logits = self._generate(
        #     input_ids=encoded_texts["input_ids"],
        #     attention_mask=encoded_texts["attention_mask"],
        #     max_new_tokens=self.max_new_tokens,
        #     temperature=self.temperature,
        #     return_logprobs=self.logprobs,
        #     top_k=self.top_logprobs,
        #     **kwargs,
        # )
        # pred_ids = completed_ids[0]
        # logprobs = logprobs[0]
        # logits = logits[0]
        # sorted_logprobs, sorted_indices = torch.sort(logprobs, descending=True)
        # sorted_logits = logits[torch.arange(logits.size(0)).unsqueeze(1), sorted_indices]
        # sorted_logprobs = sorted_logprobs[:, :self.top_logprobs]
        # sorted_logits = sorted_logits[:, :self.top_logprobs]
        # sorted_indices = sorted_indices[:, :self.top_logprobs]
        # # Decode the response
        # answer = self.tokenizer.decode(pred_ids, skip_special_tokens=True)
        # logprobs = [[(self.id_to_token(token_id), logprob.item(), logit.item()) for token_id, logprob, logit in zip(top_indices, top_logprobs, top_logits)] for top_indices, top_logprobs, top_logits in zip(sorted_indices, sorted_logprobs, sorted_logits)]
        # return answer, logprobs


if __name__ == "__main__":
    # Example usage
    model = vLLM(model_name_or_path="google/shieldgemma-9b")
    prompt = "What is the capital of France?"
    outputs = model._complete(prompt)
    print(outputs)