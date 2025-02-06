import os
from datasets import load_dataset


if os.path.exists("./datasets") is False:
    os.mkdir("./datasets")

if os.path.exists("./datasets/jailbreak_bench") is False:
    os.mkdir("./datasets/jailbreak_bench")
    load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", cache_dir="./datasets/jailbreak_bench")
    load_dataset("JailbreakBench/JBB-Behaviors", "judge_comparison", cache_dir="./datasets/jailbreak_bench")

if os.path.exists("./datasets/wildguardmix") is False:
    os.mkdir("./datasets/wildguardmix")
    load_dataset("allenai/wildguardmix", "wildguardtrain", cache_dir="./datasets/wildguardmix")
    load_dataset("allenai/wildguardmix", "wildguardtest", cache_dir="./datasets/wildguardmix")

if os.path.exists("./datasets/aegis_ai_content-safety") is False:
    os.mkdir("./datasets/aegis_ai_content-safety")
    load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-1.0", cache_dir="./datasets/aegis_ai_content_safety")