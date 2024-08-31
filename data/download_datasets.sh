git clone https://github.com/orhonovich/instruction-induction.git instruction_induction
git clone https://github.com/allenai/natural-instructions.git natural_instructions
git clone https://github.com/JailbreakBench/artifacts.git jailbreak_bench

mkdir datasets
mv instruction_induction datasets/instruction_induction
mv natural_instructions datasets/natural_instructions
mv jailbreak_bench datasets/jailbreak_bench

python download_huggingface_datasets.py