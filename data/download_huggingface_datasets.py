from datasets import load_dataset


load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", cache_dir="./datasets/jailbreak_bench")
load_dataset("JailbreakBench/JBB-Behaviors", "judge_comparison", cache_dir="./datasets/jailbreak_bench")

load_dataset("allenai/wildguardmix", "wildguardtrain", cache_dir="./data/datasets/wildguardmix")
load_dataset("allenai/wildguardmix", "wildguardtest", cache_dir="./data/datasets/wildguardmix")