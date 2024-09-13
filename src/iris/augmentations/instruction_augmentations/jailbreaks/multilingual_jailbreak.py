import logging
import requests
from iris.cache import CacheStorage
from typing import List, Tuple, Callable
from easyjailbreak.models import ModelBase
from easyjailbreak.mutation.rule import Translate
from easyjailbreak.datasets import JailbreakDataset, Instance
from iris.augmentations.instruction_augmentations.jailbreaks import Jailbreaking

logger = logging.getLogger()
logger.setLevel(logging.WARNING)


class CacheableTranslate(Translate):
    def __init__(self, attr_name='query', language='en', use_cache: bool = True, cache_path: str = None):
        self.use_cache = use_cache
        self.cache_storage = CacheStorage(f"easyjailbreak/mutation/translate_{language}", cache_path)
        super().__init__(
            attr_name=attr_name, 
            language=language
        )

    def translate(self, text, src_lang='auto'):
        if self.use_cache:
            translation = self.cache_storage.retrieve(text)
        if translation is None:
            translation = super().translate(text, src_lang)
            self.cache_storage.cache(translation, text)
        return translation


class MultiLingualJailbreaking(Jailbreaking):
    def __init__(
        self, 
        target_model: ModelBase, 
        evaluator: Callable = None,
        include_failed_cases: bool = False,
        apply_jailbreak_template: bool = True,
        translate_answer_to_en: bool = True,
        use_cache: bool = True,
        cache_path: str = None,
        **kwargs,
    ):
        if not include_failed_cases and evaluator is None:
            raise ValueError("Evaluator must be provided if include_failed_cases is False")
        
        self.target_model = target_model
        self.evaluator = evaluator
        self.apply_jailbreak_template = apply_jailbreak_template
        self.translate_answer_to_en = translate_answer_to_en

        self.mutations = [
            # Chinese
            CacheableTranslate(language='zh-CN', use_cache=use_cache, cache_path=cache_path),
            # Italian
            CacheableTranslate(language='it', use_cache=use_cache, cache_path=cache_path),
            # Vietnamese
            CacheableTranslate(language='vi', use_cache=use_cache, cache_path=cache_path),
            # Arabic
            CacheableTranslate(language='ar', use_cache=use_cache, cache_path=cache_path),
            # Korean
            CacheableTranslate(language='ko', use_cache=use_cache, cache_path=cache_path),
            # Thai
            CacheableTranslate(language='th', use_cache=use_cache, cache_path=cache_path),
            # Bengali
            CacheableTranslate(language='bn', use_cache=use_cache, cache_path=cache_path),
            # Swahili
            CacheableTranslate(language='sw', use_cache=use_cache, cache_path=cache_path),
            # Javanese
            CacheableTranslate(language='jv', use_cache=use_cache, cache_path=cache_path),
        ]
        super().__init__(include_failed_cases=include_failed_cases)

    @staticmethod
    def _translate_to_en(text, src_lang='auto'):
        """
        Translate target response to English.
        Credit: https://github.com/EasyJailbreak/EasyJailbreak/blob/master/easyjailbreak/attacker/Multilingual_Deng_2023.py
        """
        googleapis_url = 'https://translate.googleapis.com/translate_a/single'
        url = '%s?client=gtx&sl=%s&tl=%s&dt=t&q=%s' % (googleapis_url,src_lang,'en',text)
        data = requests.get(url).json()
        res = ''.join([s[0] for s in data[0]])
        return res

    def _single_attack(self, instruction: str, reference_answers: List[str] = None) -> List[Tuple[str, str]]:
        """
        Perform a Multilingual single attack on the target model.
        Credit: https://github.com/EasyJailbreak/EasyJailbreak/blob/master/easyjailbreak/attacker/Multilingual_Deng_2023.py
        """
        instance_dataset = JailbreakDataset([Instance(query=instruction, target_responses=reference_answers)])

        mutated_instance_list = []
        for mutation in self.mutations:
            transformed_dataset = mutation(instance_dataset)
            for item in transformed_dataset:
                mutated_instance_list.append(item)

        attack_results: List[Tuple[str, str]] = []
        for instance in mutated_instance_list:
            if self.apply_jailbreak_template and instance.jailbreak_prompt is not None:
                jailbreak_prompt = instance.jailbreak_prompt.format(translated_query=instance.translated_query)
            else:
                jailbreak_prompt = instance.query
            target_response = self.target_model.generate(jailbreak_prompt)
            target_response = self._translate_to_en(target_response) if self.translate_answer_to_en else target_response
            attack_results.append((jailbreak_prompt, target_response))
        return attack_results

    def augment(self, instruction: str, reference_answers: List[str] = None) -> List[str]:
        attack_results = self._single_attack(instruction=instruction, reference_answers=reference_answers)
        if not self.include_failed_cases:
            jailbreak_prompts = [jailbreak_prompt for jailbreak_prompt, response in attack_results if self.evaluator(response)]
        else:
            jailbreak_prompts = [jailbreak_prompt for jailbreak_prompt, _ in attack_results]
        return jailbreak_prompts


if __name__ == "__main__":
    from iris.datasets import JailbreakBenchDataset
    from iris.model_wrappers.guard_models import WildGuard

    target_model = WildGuard(
        model_name_or_path="allenai/wildguard",
        api_key="EMPTY",
        api_base="http://10.204.100.70:11699/v1",
        cache_path="./cache",
        use_cache=True,
    )
    def evaluator(text: str) -> bool:
        return text.strip().capitalize() == "Benign"
    
    harmful_dataset = JailbreakBenchDataset(intention="harmful")
    harmful_samples = harmful_dataset.as_samples()
    
    # Test without jailbreak_template
    augmentation = MultiLingualJailbreaking(
        target_model=target_model, 
        evaluator=evaluator,
        apply_jailbreak_template=False,
        translate_answer_to_en=False,
        cache_path="./cache",
    )

    jailbreaked_samples = augmentation.augment_batch(harmful_samples)
    print(f"ASR on Harmful prompts with MultiLingualJailbreaking: {augmentation.attack_success_rate}")

    # Test without jailbreak_template
    augmentation = MultiLingualJailbreaking(
        target_model=target_model, 
        evaluator=evaluator,
        apply_jailbreak_template=True,
        translate_answer_to_en=False,
        cache_path="./cache",
    )
    jailbreaked_samples = augmentation.augment_batch(harmful_samples)
    print(f"ASR on Harmful prompts with MultiLingualJailbreaking + jailbreak template: {augmentation.attack_success_rate}")

    harmful_dataset = JailbreakBenchDataset(intention="benign")
    harmful_samples = harmful_dataset.as_samples()
    
    # Test without jailbreak_template
    augmentation = MultiLingualJailbreaking(
        target_model=target_model, 
        evaluator=evaluator,
        apply_jailbreak_template=False,
        translate_answer_to_en=False,
        cache_path="./cache",
    )

    jailbreaked_samples = augmentation.augment_batch(harmful_samples)
    print(f"ASR on Benign prompts with MultiLingualJailbreaking: {augmentation.attack_success_rate}")

    # Test without jailbreak_template
    augmentation = MultiLingualJailbreaking(
        target_model=target_model, 
        evaluator=evaluator,
        apply_jailbreak_template=True,
        translate_answer_to_en=False,
        cache_path="./cache",
    )
    jailbreaked_samples = augmentation.augment_batch(harmful_samples)
    print(f"ASR on Benign prompts with MultiLingualJailbreaking + jailbreak template: {augmentation.attack_success_rate}")