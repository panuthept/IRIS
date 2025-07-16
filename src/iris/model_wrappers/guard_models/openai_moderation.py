from typing import Optional
from iris.data_types import SafeGuardInput, SafeGuardResponse


class OpenAIModeration:
    def __init__(
            self, 
            api_key: str,
            model_name: str = "omni-moderation-latest",
            **kwargs,
    ):
        from openai import OpenAI
        
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def parse_label(self, outputs: 'ModerationCreateResponse'):
        result = outputs.results[0]

        max_confidence = 0.0
        for category, score in list(result.category_scores):
            if score > max_confidence:
                max_confidence = score

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
        prompt_gold_label = None
        response_gold_label = None
        if input is not None:
            prompt = input.prompt
            response = input.response
            prompt_gold_label = input.prompt_gold_label
            response_gold_label = input.response_gold_label
        assert prompt is not None, "Prompt cannot be None"
        # Initial metadata
        metadata = {}
        # Prompt classification
        outputs = self.client.moderations.create(
            model=self.model_name,
            input=prompt,
        )
        try:
            answer, logprobs = self.parse_label(outputs)
        except Exception as e:
            print(f"Error parsing response: {e}")
            print(outputs)
            answer = "Safe"
            logprobs = [("Safe", 1.0, None), ("Harmful", 0.0, None)]
        # Response classification
        prompt_labels = logprobs
        metadata["prompt_instruction"] = prompt
        metadata["prompt_response"] = answer
        # Output formatting
        output = SafeGuardResponse(
            prompt=prompt, 
            response=response,
            prompt_gold_label=prompt_gold_label,
            response_gold_label=response_gold_label,
            prompt_labels=prompt_labels,
            response_labels=None,
            metadata=metadata,
        )
        return output
    

if __name__ == "__main__":
    model = OpenAIModeration(
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