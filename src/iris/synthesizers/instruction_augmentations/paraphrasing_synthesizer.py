from iris.synthesizers.instruction_augmentations.base import TextSynthesizer


class ParaphrasingSynthesizer(TextSynthesizer):
    """Synthesizer that generates misspelled text."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.instruction = (
            "Generate a parapgrased version of the input text.\n"
        )
    

if __name__ == "__main__":
    from llama_index.llms.openai import OpenAI

    gen = ParaphrasingSynthesizer(
        llm=OpenAI(
            model="gpt-3.5-turbo", 
            api_key="sk-proj-uvbi9yfICRLlEdB9WuVLT3BlbkFJLI51rD9gebE9T5pxxztV",
        ),
    )
    resp = gen._synthesize("The quick brown fox jumps over the lazy dog.")
    print(resp)
    print(type(resp))