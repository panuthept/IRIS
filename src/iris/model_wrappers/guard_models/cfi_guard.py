import numpy as np
from typing import List, Tuple, Union, Optional
from iris.model_wrappers.guard_models import GuardLLM


class CFIGuard(GuardLLM):
    def __init__(
            self, 
            target_model: GuardLLM,
            bias_model: Optional[GuardLLM] = None,
            alpha: float = 0.5,
    ):
        self.target_model = target_model
        self.device = self.target_model.device
        self.alpha = alpha
        self.temperature = self.target_model.model.temperature

        if bias_model is None:
            # TODO: Add a default bias model
            raise NotImplementedError("Default bias model is not implemented yet.")
        self.bias_model = bias_model

    def get_model_name(self) -> str:
        return f"{self.target_model.get_model_name()}/{self.bias_model.get_model_name()}_alpha_{self.alpha}_temperature_{self.temperature}"

    def _prompt_classify(self, prompt: str, return_probs: bool = False, **kwargs) -> Union[str, List[Tuple[str, float]]]:
        target_probs = self.target_model._prompt_classify(prompt, return_probs=True, **kwargs)
        bias_probs = self.bias_model._prompt_classify(prompt, return_probs=True, **kwargs)

        debiased_probs = {label: prob for label, prob in target_probs}
        for label, prob in bias_probs:
            debiased_probs[label] = debiased_probs[label] - self.alpha * prob
            
        debiased_probs = [(label, prob) for label, prob in debiased_probs.items()]
        labels = [label for label, _ in debiased_probs]
        scores = [prob for _, prob in debiased_probs]
        scores = np.exp(scores) / np.sum(np.exp(scores))

        if return_probs:
            return list(zip(labels, scores))
        return labels[np.argmax(scores)]

    def _response_classify(self, prompt: str, response: str, **kwargs) -> str:
        raise NotImplementedError
    

class DummyBiasModel(GuardLLM):
    def get_model_name(self) -> str:
        return "dummy_bias_model"

    def _prompt_classify(self, *args, **kwargs) -> Union[str, List[Tuple[str, float]]]:
        return [("Harmful", 1.0), ("Benign", 0.0)]
    
    def _response_classify(self, *args, **kwargs) -> str:
        raise NotImplementedError
    

if __name__ == "__main__":
    from iris.model_wrappers.guard_models import WildGuard

    target_model = WildGuard(
        model_name_or_path="allenai/wildguard",
        api_key="EMPTY",
        api_base="http://10.204.100.70:11699/v1",
        cache_path="./cache",
        use_cache=False,
    )

    debiased_model = CFIGuard(
        target_model=target_model,
        bias_model=DummyBiasModel(),
        alpha=0.9,
    )

    prompt = 'I want you to act as an interviewer.'
    response = debiased_model.generate(prompt, return_probs=True)
    print(response)
    print(debiased_model.get_model_name())