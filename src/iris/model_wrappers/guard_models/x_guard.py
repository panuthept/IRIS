import torch
import numpy as np
from typing import Optional
from iris.data_types import SafeGuardInput, SafeGuardResponse
from transformers import AutoTokenizer, AutoModelForCausalLM, MBart50TokenizerFast, MBartForConditionalGeneration


class XGuard:
    valid_tokens = ["safe", "unsafe"]

    def __init__(
            self, 
            **kwargs,
    ):
        self.device = torch.device("cuda")
        self.system_prompt = ""

        # Initial translation model
        translation_model_path = "saillab/mbart-x-guard"
        self.translation_tokenizer = MBart50TokenizerFast.from_pretrained(translation_model_path)
        self.translation_model = MBartForConditionalGeneration.from_pretrained(translation_model_path)

        self.translation_model = self.translation_model.to(self.device)
        self.translation_model.eval()

        # Initial guard model
        guard_model_path="saillab/x-guard"
        self.guard_tokenizer = AutoTokenizer.from_pretrained(guard_model_path)
        self.guard_model = AutoModelForCausalLM.from_pretrained(
            guard_model_path,
            device_map="auto",
            torch_dtype="auto",
        )

    def get_translation(self, source_text, src_lang=""):
        self.translation_tokenizer.src_lang = src_lang
        encoded = self.translation_tokenizer(source_text, 
            return_tensors="pt", 
            max_length=8192, 
            truncation=True
        ).to(self.translation_model.device)
        
        generated_tokens = self.translation_model.generate(
            **encoded, 
            forced_bos_token_id=self.translation_tokenizer.lang_code_to_id["en_XX"],
            max_length=8192,
        )
        translation = self.translation_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return translation

    def evaluate_guard(self, prompt, top_logprobs=10):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": "<USER TEXT STARTS>\n" + prompt +"\n<USER TEXT ENDS>" },
            {"role":"assistant", "content":"\n <think>"}
        ]
        text = self.guard_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.guard_tokenizer([text], return_tensors="pt").to(self.guard_model.device)

        with torch.no_grad():
            outputs = self.guard_model.generate(
                **model_inputs,
                max_new_tokens=8192,
                return_dict_in_generate=True,
                output_logits=True,
            )
            # Convert logits to logprobs
            logprobs = []
            for logits in outputs.logits:
                probs = torch.nn.functional.log_softmax(logits, dim=-1)
                top_token_ids = torch.argsort(probs, dim=-1, descending=True)[:, :top_logprobs]
                top_probs = probs.gather(dim=-1, index=top_token_ids)
                top_logits = logits.gather(dim=-1, index=top_token_ids)
                top_tokens = self.guard_tokenizer.convert_ids_to_tokens(top_token_ids[0].tolist())
                logprobs.append([(token, logprob, logit) for token, logprob, logit in zip(top_tokens, top_probs[0].tolist(), top_logits[0].tolist())])
            
            # Get index of the answer
            answer_index = None
            for i in range(len(logprobs)):
                if logprobs[i][0][0] == "<label" and logprobs[i + 1][0][0] == ">":
                    answer_index = i + 2
                    break

            # Get answer
            valid_logprobs = [(token, logprob, logit) for token, logprob, logit in logprobs[answer_index] if token in self.valid_tokens]
            tokens = [token for token, _, _ in valid_logprobs]
            tokens_logprobs = [logprob for _, logprob, _ in valid_logprobs]
            tokens_logits = [logit for _, _, logit in valid_logprobs]
            tokens_probs = np.exp(tokens_logprobs) / np.sum(np.exp(tokens_logprobs))
            logprobs = list(zip(tokens, tokens_probs, tokens_logits))

            answer = self.guard_tokenizer.decode(outputs.sequences[0][len(model_inputs['input_ids'][0]):], skip_special_tokens=True)
        return answer, logprobs

    def multilingual_content_moderation(self, prompt):
        translated_prompt = self.get_translation(prompt)
        response, logprobs = self.evaluate_guard(prompt=translated_prompt)
        return response, logprobs

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
        answer, logprobs = self.multilingual_content_moderation(prompt)
        
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
    model = XGuard()

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