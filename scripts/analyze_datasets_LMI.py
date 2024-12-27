import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer
from iris.datasets import load_dataset
from sklearn.linear_model import LinearRegression


if __name__ == "__main__":
    topk = 100000
    threshold = 0.0

    with open("./data/LMI_scores.jsonl", "r") as f:
        data = json.load(f)
        tokenizer = AutoTokenizer.from_pretrained(data["tokenizer"])
        # harmful_LMIs = {k: v for k, v in data["LMI"]["harmful"].items() if v > threshold}
        harmful_LMIs = {k: v for k, v in data["LMI"]["harmful"].items()}
        train_harmful_LMIs = [(k, v) for k, v in data["LMI"]["harmful"].items() if v > threshold]
        train_unharmful_LMIs = [(k, v) for k, v in data["LMI"]["benign"].items() if v > threshold]
        print(f"Train harmful LMIs: {len(train_harmful_LMIs)} / {len(data['LMI']['harmful'])}")
        print(f"Train unharmful LMIs: {len(train_unharmful_LMIs)} / {len(data['LMI']['benign'])}")
        # Get top-k
        if topk is not None:
            train_harmful_LMIs = train_harmful_LMIs[:topk]
            train_unharmful_LMIs = train_unharmful_LMIs[:topk]
        # Convert string to int
        harmful_LMIs = {int(k): v for k, v in harmful_LMIs.items()}
        train_harmful_LMIs = set([int(k) for k, v in train_harmful_LMIs])
        train_unharmful_LMIs = set([int(k) for k, v in train_unharmful_LMIs])

    plt.figure(figsize=(15, 3.2))
    output_path = "./outputs/WildGuard"
    dataset_names = [
        "WildGuardMixDataset",
        "ORBenchDataset",
        "OpenAIModerationDataset",
        "ToxicChatDataset",
        "XSTestDataset",
        "JailbreakBenchDataset",
    ]
    for i, dataset_name in enumerate(dataset_names):
        Xs = []
        Ys = []
        Cs = []
        statistic = {"Harmful": {}, "Benign": {}}
        dataset_path = output_path + "/" + dataset_name + "/test/all_prompts.jsonl"
        print(f"{dataset_path}:")
        with open(dataset_path, "r") as f:
            for line in f:
                example = json.loads(line)
                prompt = example["prompt"]
                pred = example["response"][0]
                pred_score = [response[1] for response in example["response"] if response[0] == "Harmful"][0]
                label = "Harmful" if example["label"] == 1 else "Benign"
                if label != "Benign":
                    continue
                # if label != "Harmful":
                #     continue
                
                sum_LMI = 0.0
                mean_LMI = []
                matched_count = {"Harmful_Shortcut": 0, "Benign_Shortcut": 0}
                prompt_words = tokenizer(prompt)
                for prompt_word in prompt_words["input_ids"]:
                    matched_count["Harmful_Shortcut"] += int(prompt_word in train_harmful_LMIs)
                    matched_count["Benign_Shortcut"] += int(prompt_word in train_unharmful_LMIs)
                    if prompt_word in harmful_LMIs:
                        sum_LMI += harmful_LMIs[prompt_word]
                        mean_LMI.append(harmful_LMIs[prompt_word])
                mean_LMI = np.mean(mean_LMI)

                # matched_ratio = 0.5
                # matched_ratio = mean_LMI
                # matched_ratio = sum_LMI
                matched_ratio = matched_count["Harmful_Shortcut"] / len(prompt_words["input_ids"])
                # if dataset_name == "XSTestDataset" and matched_ratio > 0.5:
                #     print(prompt)
                #     print(pred_score)
                #     print(matched_ratio)
                #     print("-" * 100)
                # matched_ratio = matched_count["Benign_Shortcut"] / len(prompt_words["input_ids"])
                # matched_ratio = matched_count["Harmful_Shortcut"]
                if matched_ratio not in statistic[label]:
                    statistic[label][matched_ratio] = {"Incorrect": 0, "Total": 0, "Scores": []}
                statistic[label][matched_ratio]["Total"] += 1
                statistic[label][matched_ratio]["Incorrect"] += int(pred[0] != label)
                statistic[label][matched_ratio]["Scores"].append(pred_score)

                # Update data for regression
                Xs.append(matched_ratio)
                # Ys.append(pred_score)
                Ys.append(int(pred[0] != label))
                Cs.append(label)

        plt.subplot(1, len(dataset_names), i + 1)

        # Calculate the statistic
        # statistic["Harmful"] = [(k, round(v["Incorrect"] / v["Total"], 2), v["Total"]) for k, v in sorted(statistic["Harmful"].items(), key=lambda x: x[0], reverse=False)]
        # # statistic["Harmful"] = [(k, round(np.mean(v["Scores"]), 2), v["Total"]) for k, v in sorted(statistic["Harmful"].items(), key=lambda x: x[0], reverse=False)]
        # xs = [x for x, y, s in statistic["Harmful"]]
        # ys = [y for x, y, s in statistic["Harmful"]]
        # ss = [s for x, y, s in statistic["Harmful"]]
        # plt.scatter(xs, ys, s=ss, alpha=0.5, c="red", label="Harmful")

        statistic["Benign"] = [(k, round(v["Incorrect"] / v["Total"], 2), v["Total"]) for k, v in sorted(statistic["Benign"].items(), key=lambda x: x[0], reverse=False)]
        # statistic["Benign"] = [(k, round(np.mean(v["Scores"]), 2), v["Total"]) for k, v in sorted(statistic["Benign"].items(), key=lambda x: x[0], reverse=False)]
        xs = [x for x, y, s in statistic["Benign"]]
        ys = [y for x, y, s in statistic["Benign"]]
        ss = [s for x, y, s in statistic["Benign"]]
        plt.scatter(xs, ys, s=ss, alpha=0.5, c="blue", label="Benign")

        # print(len([y for y in Ys if y < 0.5]))

        # plt.scatter([x for x, c in zip(Xs, Cs) if c == "Harmful"], [y for y, c in zip(Ys, Cs) if c == "Harmful"], s=1, alpha=0.5, c="red", label="Harmful")
        # plt.scatter([x for x, c in zip(Xs, Cs) if c == "Benign"], [y for y, c in zip(Ys, Cs) if c == "Benign"], s=1, alpha=0.5, c="blue", label="Benign")

        # Linear regression
        linear_model = LinearRegression()
        xs = np.array(Xs).reshape(-1, 1)
        ys = np.array(Ys)
        linear_model.fit(xs, ys)
        print(f"Linear model: {linear_model.coef_}, {linear_model.intercept_}")
        plt.axline(xy1=(0, linear_model.intercept_), slope=linear_model.coef_[0], c="black", linestyle="--")
        # zs = linear_model.predict(xs)
        # plt.plot(xs, zs, c="black", label=dataset_name.replace("Dataset", ""))

        plt.title(dataset_name.replace("Dataset", "") + "\n" + f"(coef: {linear_model.coef_[0]:.2f})")
        plt.xlabel("Shortcut Token Ratio")
        # plt.xlabel("Benign Shortcut Ratio")
        plt.ylim(-0.05, 1.05)
        plt.xlim(np.min(xs) - 0.05, np.max(xs) + 0.05)

        if i == 0:
            plt.ylabel("Exaggerated Safety")
    plt.tight_layout()
    plt.show()