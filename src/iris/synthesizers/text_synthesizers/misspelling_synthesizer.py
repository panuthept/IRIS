from llama_index.llms.openai import OpenAI
from iris.synthesizers.text_synthesizers.base import BaseTextSynthesizer


class MisspellingSynthesizer(BaseTextSynthesizer):
    """Synthesizer that generates misspelled text."""
    def __init__(self, *args, max_misspelled_nums: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.instruction = (
            "Generate a misspelled version of the input text.\n"
            "Maximum number of misspelled words: {max_misspelled_nums}."
        ).format(max_misspelled_nums=max_misspelled_nums)
    

if __name__ == "__main__":
    gen = MisspellingSynthesizer(
        llm=OpenAI(
            model="gpt-3.5-turbo", 
            api_key="sk-proj-uvbi9yfICRLlEdB9WuVLT3BlbkFJLI51rD9gebE9T5pxxztV",
        ),
        max_misspelled_nums=2,
    )
    resp = gen.synthesize("The quick brown fox jumps over the lazy dog.")
    print(resp)
    print(type(resp))