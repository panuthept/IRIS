from datasets import load_dataset


load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", cache_dir="./datasets/jailbreak_bench")
load_dataset("JailbreakBench/JBB-Behaviors", "judge_comparison", cache_dir="./datasets/jailbreak_bench")