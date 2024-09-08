from iris.augmentations.instruction_augmentations.base import TextSynthesizer


class MisspellingSynthesizer(TextSynthesizer):
    """Synthesizer that generates misspelled text."""
    def __init__(self, *args, max_misspelled_nums: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.instruction = (
            "Generate a misspelled version of the input text.\n"
            "Maximum number of misspelled words: {max_misspelled_nums}."
        ).format(max_misspelled_nums=max_misspelled_nums)
    

if __name__ == "__main__":
    from llama_index.llms.openai import OpenAI

    gen = MisspellingSynthesizer(
        llm=OpenAI(
            model="gpt-3.5-turbo", 
            api_key="sk-proj-uvbi9yfICRLlEdB9WuVLT3BlbkFJLI51rD9gebE9T5pxxztV",
        ),
        max_misspelled_nums=2,
    )
    resp = gen._synthesize("The quick brown fox jumps over the lazy dog.")
    print(resp)
    print(type(resp))