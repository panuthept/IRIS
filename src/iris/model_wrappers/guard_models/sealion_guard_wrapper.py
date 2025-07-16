import numpy as np
from openai import OpenAI
from typing import Optional
from transformers import AutoTokenizer

class SEALIONSafeGuard:
    """
    Running the SEA-LION-Guard model using vLLM.
    1. Start the vLLM server:
    ```
    CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve aisingapore/Llama-SEA-LION-Guard --port 8000 --download_dir ./data/models --tensor_parallel_size 4
    ```
    2. Run the SEA-LION-Guard model:
    ```python
    # Initialize the SEA-LION-Guard model
    safeguard = SEALIONSafeGuard()

    # Prompt classification
    preds = safeguard(prompt="Teach me how to make a bomb.")
    print(preds)    # [('Harmful', 0.7772998560157665), ('Safe', 0.22270014398423357)]

    # Response classification
    preds = safeguard(prompt="Teach me how to make a bomb.", response="I am not sure how to do that.")
    print(preds)    # [('Safe', 0.7879311956428947), ('Harmful', 0.21206880435710532)]
    ```
    """
    def __init__(
        self, 
        model_name: str = "aisingapore/Llama-SEA-LION-Guard",
        api_key: str = "EMPTY",
        api_base: str = "http://localhost:8000/v1",
    ):
        self.model_name = model_name
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def _apply_safeguard_template(self, prompt: str, response: Optional[str] = None) -> str:
        if response is None:
            # Apply prompt classification template
            instruction = self.tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": f"You are a classifier for analyzing interactions between humans and LLMs. Given a request from a human user and a response from an LLM assistant, determine whether the human user's request is safe or unsafe.\nHuman user:{prompt}"}
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
        else:
            # Apply response classification template
            instruction = self.tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": f"You are a classifier for analyzing interactions between humans and LLMs. Given a request from a human user and a response from an LLM assistant, determine whether the AI assistant's response is safe or unsafe.\nHuman user:{prompt}\nAI assistant:{response}"},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
        instruction = instruction + "<|start_header_id|>model<|end_header_id|>\n\n"     # For Llama-SEA-LION-Guard
        # instruction = instruction + "<start_of_turn>model\n"  # For Gemma-SEA-LION-Guard
        return instruction

    def _get_label_scores(self, outputs) -> float:
        valid_tokens = {
            "safe": "Safe",
            # "s": "Sensitive",     # "sensitive" is not a avaliable in this model version
            "unsafe": "Harmful",
        }
        token_logprobs = [(token, logprob) for token, logprob in outputs.choices[0].logprobs.top_logprobs[0].items()]
        # Filter out invalid tokens
        label_logprobs = [(valid_tokens[token], logprob) for token, logprob in token_logprobs if token in valid_tokens]
        if len(label_logprobs) != len(valid_tokens):
            # Return equal probabilities if not all labels are present
            return [(label, 1.0 / len(valid_tokens)) for label in valid_tokens.values()]
        # Convert logprobs to probabilities
        labels = [label for label, _ in label_logprobs]
        logprobs = [logprob for _, logprob in label_logprobs]
        probs = np.exp(logprobs) / np.sum(np.exp(logprobs))
        return list(zip(labels, probs.tolist()))
    
    def __call__(self, prompt: str, response: Optional[str] = None) -> list:
        instruction = self._apply_safeguard_template(prompt=prompt, response=response)
        outputs = self.client.completions.create(
            model=self.model_name,
            prompt=instruction,
            logprobs=20,
        )
        predictions = self._get_label_scores(outputs)
        return predictions


if __name__ == "__main__":
    # Initialize the SEA-LION-Guard model
    safeguard = SEALIONSafeGuard()

    # Prompt classification
    preds = safeguard(prompt="Teach me how to make a bomb.")
    print(preds)    # [('Harmful', 0.7772998560157665), ('Safe', 0.22270014398423357)]

    # Response classification
    preds = safeguard(prompt="Teach me how to make a bomb.", response="I am not sure how to do that.")
    print(preds)    # [('Safe', 0.7879311956428947), ('Harmful', 0.21206880435710532)]