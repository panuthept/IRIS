from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict


@dataclass
class PromptTemplate:
    instruction_template: str = "Instruction: {instruction}\nOutput: "
    instruction_query_template: str = "Instruction: {instruction}\nInput: {query}\nOutput: "
    instruction_examples_query_template: str = "Instruction: {instruction}\n{examples}\nInput: {query}\nOutput: "
    query_template: str = "Input: {query}\nOutput: "
    query_examples_template: str = "{examples}\nInput: {query}\nOutput: "
    instruction_answer_template: str = "Instruction: {instruction}\nOutput: {answer}"
    instruction_query_answer_template: str = "Instruction: {instruction}\nInput: {query}\nOutput: {answer}"
    instruction_examples_query_answer_template: str = "Instruction: {instruction}\n{examples}\nInput: {query}\nOutput: {answer}"
    query_answer_template: str = "Input: {query}\nOutput: {answer}"
    query_examples_answer_template: str = "{examples}\nInput: {query}\nOutput: {answer}"

    def as_dict(self) -> Dict[str, str]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> 'PromptTemplate':
        # Filter out None values and unknown keys
        data = {k: v for k, v in data.items() if v is not None and k in cls.__annotations__}
        return cls(**data)
    
    def get_example_prompt(self, examples: List[Tuple[str, str]] = None) -> str:
        prompts = []
        for query, answer in examples:
            prompts.append(self.query_answer_template.format(query=query, answer=answer))
        return "\n".join(prompts)

    def _get_prompt(
        self,
        query: str = None,
        instruction: str = None,
        examples: List[Tuple[str, str]] = None,
        answer: str = None,
    ):
        if query:
            if instruction:
                if examples:
                    if answer:
                        name = "instruction_examples_query_answer_template"
                        template = self.instruction_examples_query_answer_template
                        prompt = template.format(
                            instruction=instruction, examples=self.get_example_prompt(examples), query=query, answer=answer
                        )
                    else:
                        name = "instruction_examples_query_template"
                        template = self.instruction_examples_query_template
                        prompt = template.format(instruction=instruction, examples=self.get_example_prompt(examples), query=query)
                else:
                    if answer:
                        name = "instruction_query_answer_template"
                        template = self.instruction_query_answer_template
                        prompt = template.format(instruction=instruction, query=query, answer=answer)
                    else:
                        name = "instruction_query_template"
                        template = self.instruction_query_template
                        prompt = template.format(instruction=instruction, query=query)
            else:
                if examples:
                    if answer:
                        name = "query_examples_answer_template"
                        template = self.query_examples_answer_template
                        prompt = template.format(examples=self.get_example_prompt(examples), query=query, answer=answer)
                    else:
                        name = "query_examples_template"
                        template = self.query_examples_template
                        prompt = template.format(examples=self.get_example_prompt(examples), query=query)
                else:
                    if answer:
                        name = "query_answer_template"
                        template = self.query_answer_template
                        prompt = template.format(query=query, answer=answer)
                    else:
                        name = "query_template"
                        template = self.query_template
                        prompt = template.format(query=query)
        else:
            if instruction:
                if examples:
                    if answer:
                        name = "instruction_examples_query_answer_template"
                        template = self.instruction_examples_query_answer_template
                        prompt = template.format(instruction=instruction, examples=self.get_example_prompt(examples), answer=answer)
                    else:
                        name = "instruction_examples_query_template"
                        template = self.instruction_examples_query_template
                        prompt = template.format(instruction=instruction, examples=self.get_example_prompt(examples))
                else:
                    if answer:
                        name = "instruction_answer_template"
                        template = self.instruction_answer_template
                        prompt = template.format(instruction=instruction, answer=answer)
                    else:
                        name = "instruction_template"
                        template = self.instruction_template
                        prompt = template.format(instruction=instruction)
            else:
                name = "instruction_template"
                template = self.instruction_template
                prompt = template.format(instruction="")
        return name, template, prompt
    
    def as_partial_dict(
        self,
        query: str = None,
        instruction: str = None,
        examples: List[Tuple[str, str]] = None,
        answer: str = None,
    ) -> Dict[str, str]:
        name, template, _ = self._get_prompt(query, instruction, examples, answer)
        return {name: template}

    def get_prompt(
        self,
        query: str = None,
        instruction: str = None,
        examples: List[Tuple[str, str]] = None,
        answer: str = None,
    ):
        _, _, prompt = self._get_prompt(query, instruction, examples, answer)
        return prompt