import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer
from iris.datasets import WildGuardMixDataset


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("allenai/wildguard")

    dataset = WildGuardMixDataset()
    samples = dataset.as_samples("train")

    total_vocab = []
    cnt = defaultdict(lambda : defaultdict(lambda:0)) 
    cnt_y = defaultdict(lambda :0)
    cnt_word = defaultdict(lambda :0)
    for sample in tqdm(samples):
        prompt = sample.instructions[0]
        label = sample.instructions_true_label[0]

        prompt_words = tokenizer(prompt)
        for prompt_word in prompt_words["input_ids"]:
            cnt[label][prompt_word] += 1
            cnt_word[prompt_word] += 1
        cnt_y[label] += 1
        total_vocab.extend(prompt_words["input_ids"])

    LMIs = []
    D = len(set(total_vocab))
    T = len(total_vocab)
    p_Y = cnt_y["Benign"]/(cnt_y["Benign"]+ cnt_y["Harmful"])
    score_unharmful_LMIs = defaultdict(lambda : 0)
    for idx,freq in tqdm(cnt["Benign"].items()):
        p_W_Y = freq/D
        p_Y_W = freq/cnt_word[idx]
        LMI = p_W_Y * np.log(p_Y_W/p_Y)
        LMIs.append(float(LMI))
        score_unharmful_LMIs[idx] = float(LMI)

    LMIs = []
    D = len(set(total_vocab))
    T = len(total_vocab)
    p_Y = cnt_y["Harmful"]/(cnt_y["Benign"]+ cnt_y["Harmful"])
    score_harmful_LMIs = defaultdict(lambda : 0)
    for idx,freq in tqdm(cnt["Harmful"].items()):
        p_W_Y = freq/D
        p_Y_W = freq/cnt_word[idx]
        LMI = p_W_Y * np.log(p_Y_W/p_Y)
        LMIs.append(float(LMI))
        score_harmful_LMIs[idx] = float(LMI)

    # Save the LMI scores
    score_harmful_LMIs = dict(sorted(score_harmful_LMIs.items(), key=lambda item: item[1], reverse=True))
    score_unharmful_LMIs = dict(sorted(score_unharmful_LMIs.items(), key=lambda item: item[1], reverse=True))
    with open("./data/LMI_scores.jsonl", "w") as f:
        data = {
            "tokenizer": "allenai/wildguard",
            "LMI": {
                "harmful": score_harmful_LMIs,
                "benign": score_unharmful_LMIs,
            }
        }
        f.write(json.dumps(data))

    sorted_score_harmful_LMIs = [(idx, tokenizer.decode([idx]), score) for idx, score in score_harmful_LMIs.items()]
    sorted_score_unharmful_LMIs = [(idx, tokenizer.decode([idx]), score) for idx, score in score_unharmful_LMIs.items()]
    print(sorted_score_harmful_LMIs[:10])
    print("-" * 100)
    print(sorted_score_harmful_LMIs[-10:])
    print("*" * 100)
    print(sorted_score_unharmful_LMIs[:10])
    print("-" * 100)
    print(sorted_score_unharmful_LMIs[-10:])