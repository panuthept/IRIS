from openai import OpenAI
from transformers import AutoTokenizer
from typing import Tuple, List, Dict, Optional
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
        top_logprobs: int = 2,
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
        prompt: str = None,
        message: List[Dict[str, str]] = None,
        **kwargs
    ) -> Tuple[str, Optional[List[List[Tuple[str, float]]]]]:
        # Generate the response
        try_count = 0
        while try_count < 10:
            # try:
            if message is not None:
                print("*" * 100)
                print(f"Message: {message}")
                print(f"model_name: {self.model_name}")
                print("*" * 100)
                outputs = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=message,
                    max_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    logprobs=self.top_logprobs > 0,
                    top_logprobs=self.top_logprobs,
                    stream=False,
                    # echo=False,
                    n=1,
                )
                print(outputs)
                answer = outputs.choices[0].message.content
                print(outputs.choices[0].logprobs)
                print(outputs.choices[0].logprobs.content)
                logprobs = [[(top_logprob.token, top_logprob.logprob, None) for top_logprob in content.top_logprobs] for content in outputs.choices[0].logprobs.content]
                print(answer)
            else:
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
                logprobs = [[(k, v, None) for k, v in logprob.items()][:self.top_logprobs] for logprob in outputs.choices[0].logprobs.top_logprobs]
            break
            # except Exception as e:
            #     print(f"Error generating response: {e}")
            #     if "Error code: 400" in str(e):
            #         answer = ""
            #         logprobs = None
            #         break
            #     try_count += 1
        return answer, logprobs


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