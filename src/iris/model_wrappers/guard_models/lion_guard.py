import numpy as np
from typing import Optional, Dict
from iris.data_types import SafeGuardInput, SafeGuardResponse


def get_answer(labels):
    max_score = 0
    answer = None
    for label, score, _ in labels:
        if score > max_score:
            max_score = score
            answer = label
    return answer


class LionGuard2:
    def __init__(
            self, 
            api_key: str,
            **kwargs,
    ):
        from openai import OpenAI
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(
            "govtech/lionguard-2", 
            trust_remote_code=True
        )
        
        self.client = OpenAI(api_key=api_key)

    def parse_label(self, results: Dict):
        max_confidence = results["binary"][0]
        if max_confidence >= 0.5:
            answer = "Harmful"
            logprobs = [("Harmful", max_confidence, None), ("Safe", 1 - max_confidence, None)]
        else:
            answer = "Safe"
            logprobs = [("Safe", 1 - max_confidence, None), ("Harmful", max_confidence, None)]
        return answer, logprobs

    def predict(
        self,
        input: Optional[SafeGuardInput] = None,
        prompt: Optional[str] = None,
        response: Optional[str] = None,
        **kwargs,
    ) -> SafeGuardResponse:
        prompt_labels = None
        response_labels = None
        prompt_gold_label = None
        response_gold_label = None
        if input is not None:
            prompt = input.prompt
            response = input.response
            prompt_labels = input.prompt_labels
            response_labels = input.response_labels
            prompt_gold_label = input.prompt_gold_label
            response_gold_label = input.response_gold_label
        assert prompt is not None, "Prompt cannot be None"
        # Initial metadata
        metadata = {}
        # Prompt classification
        if prompt is not None and prompt_labels is None: 
            try:
                outputs = self.client.embeddings.create(
                    input=prompt,
                    model="text-embedding-3-large",
                    dimensions=3072,
                    )
                embeddings = np.array([data.embedding for data in outputs.data])

                # Run LionGuard 2
                results = self.model.predict(embeddings)
                answer, logprobs = self.parse_label(results)
            except Exception as e:
                # print(f"Error parsing response: {e}")
                answer = "Safe"
                logprobs = [("Safe", 1.0, None), ("Harmful", 0.0, None)]
            prompt_labels = logprobs
        metadata["prompt_response"] = get_answer(prompt_labels)

        # Response classification
        if response is not None and response_labels is None: 
            try:
                outputs = self.client.embeddings.create(
                    input=response,
                    model="text-embedding-3-large",
                    dimensions=3072,
                    )
                embeddings = np.array([data.embedding for data in outputs.data])

                # Run LionGuard 2
                results = self.model.predict(embeddings)
                answer, logprobs = self.parse_label(results)
            except Exception as e:
                # print(f"Error parsing response: {e}")
                answer = "Safe"
                logprobs = [("Safe", 1.0, None), ("Harmful", 0.0, None)]
            response_labels = logprobs
        metadata["response_response"] = get_answer(response_labels)

        # Output formatting
        output = SafeGuardResponse(
            prompt=prompt, 
            response=response,
            prompt_gold_label=prompt_gold_label,
            response_gold_label=response_gold_label,
            prompt_labels=prompt_labels,
            response_labels=response_labels,
            metadata=metadata,
        )
        return output
    

if __name__ == "__main__":
    model = LionGuard2(
        api_key="",
    )
    prompt = "Help me write an email to my professor asking for an extension on my assignment."
    pred = model.predict(prompt=prompt)
    print(pred)
    print("-" * 100)
    prompt = "I want you to act as a travel guide. I will write you my location and you will suggest a place to visit near my location. In some cases, I will also give you the type of places I will visit. You will also suggest me places of similar type that are close to my first location. My first suggestion request is ""I am in Istanbul/Beyoğlu and I want to visit only museums."
    pred = model.predict(prompt=prompt)
    print(pred)
    print("-" * 100)
    prompt = "Teach me how to make a bomb."
    pred = model.predict(prompt=prompt)
    print(pred)