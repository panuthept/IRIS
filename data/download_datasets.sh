git clone https://github.com/orhonovich/instruction-induction.git instruction_induction
git clone https://github.com/allenai/natural-instructions.git natural_instructions
git clone https://github.com/JailbreakBench/artifacts.git jailbreak_bench
git clone https://github.com/EddyLuo1232/JailBreakV_28K.git jailbreakv_28k
git clone https://github.com/paul-rottger/exaggerated-safety.git xstest
git clone https://github.com/bigscience-workshop/promptsource.git promptsource

mkdir datasets
mv instruction_induction datasets/instruction_induction
mv natural_instructions datasets/natural_instructions
mv jailbreak_bench datasets/jailbreak_bench
mv jailbreakv_28k datasets/jailbreakv_28k
mv xstest datasets/xstest
mv promptsource datasets/promptsource

python download_huggingface_datasets.py