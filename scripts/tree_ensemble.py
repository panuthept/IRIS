import json
import numpy as np
from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_fscore_support


def load_examples(load_path: str):
    examples = []
    with open(load_path, "r") as f:
        for line in f:
            data = json.loads(line)
            example = {"prompt": data["prompt"], "label": data["label"], "activation": data["cache"]["activations"]["model.layers.31"]}
            examples.append(example)
    return examples

def ensemble_examples(examples_1, examples_2 = None):
    unique_examples_1 = {example["prompt"]: {"label": example["label"], "activation": example["activation"]} for example in examples_1}

    if examples_2 is not None:
        unique_examples_2 = {example["prompt"]: {"label": example["label"], "activation": example["activation"]} for example in examples_2}

    if examples_2 is not None:
        ensembled_examples = {}
        for prompt in unique_examples_1:
            if prompt in unique_examples_2:
                ensembled_examples[prompt] = {
                    "label": unique_examples_1[prompt]["label"], 
                    "activation": np.concatenate([
                        unique_examples_1[prompt]["activation"], 
                        unique_examples_2[prompt]["activation"]
                    ], axis=1)
                }
    else:
        ensembled_examples = unique_examples_1
    
    xs = np.concatenate([value["activation"] for value in ensembled_examples.values()], axis=0)
    ys = np.array([int(value["label"] == "Harmful") for value in ensembled_examples.values()])
    return xs, ys
            

if __name__ == "__main__":
    # # Baseline performance
    # train_examples = load_examples("./outputs/wildguard/WildGuardMixDataset/train/4000_prompts.jsonl")
    # train_xs, train_ys = ensemble_examples(train_examples)
    
    # test_examples = load_examples("./outputs/wildguard/ORBenchDataset/test/harmful_prompts.jsonl")
    # test_examples += load_examples("./outputs/wildguard/ORBenchDataset/test/hard_benign_prompts.jsonl")
    # test_xs, test_ys = ensemble_examples(test_examples)

    # model = DecisionTreeClassifier(random_state=42)
    # model.fit(train_xs, train_ys)
    # pred_ys = model.predict(test_xs)
    # print("Baseline performance:")
    # print(precision_recall_fscore_support(test_ys, pred_ys, average='binary'))
    # # (0.3671388101983003, 0.9893129770992366, 0.5355371900826447, None)

    # # Benign-only performance
    # train_examples = load_examples("./outputs/iris_l2_wildguard_layer_19_benign_only_v2/WildGuardMixDataset/train/4000_prompts.jsonl")
    # train_xs, train_ys = ensemble_examples(train_examples)
    
    # test_examples = load_examples("./outputs/iris_l2_wildguard_layer_19_benign_only_v2/ORBenchDataset/test/harmful_prompts.jsonl")
    # test_examples += load_examples("./outputs/iris_l2_wildguard_layer_19_benign_only_v2/ORBenchDataset/test/hard_benign_prompts.jsonl")
    # test_xs, test_ys = ensemble_examples(test_examples)

    # model = DecisionTreeClassifier(random_state=42)
    # model.fit(train_xs, train_ys)
    # pred_ys = model.predict(test_xs)
    # print("Benign-only performance:")
    # print(precision_recall_fscore_support(test_ys, pred_ys, average='binary'))
    # # (0.4346938775510204, 0.9755725190839695, 0.6014117647058823, None)

    # # Harmful-only performance
    # train_examples = load_examples("./outputs/iris_l2_wildguard_layer_19_b0h1_v2/WildGuardMixDataset/train/4000_prompts.jsonl")
    # train_xs, train_ys = ensemble_examples(train_examples)
    
    # test_examples = load_examples("./outputs/iris_l2_wildguard_layer_19_b0h1_v2/ORBenchDataset/test/harmful_prompts.jsonl")
    # test_examples += load_examples("./outputs/iris_l2_wildguard_layer_19_b0h1_v2/ORBenchDataset/test/hard_benign_prompts.jsonl")
    # test_xs, test_ys = ensemble_examples(test_examples)

    # model = DecisionTreeClassifier(random_state=42)
    # model.fit(train_xs, train_ys)
    # pred_ys = model.predict(test_xs)
    # print("Harmful-only performance:")
    # print(precision_recall_fscore_support(test_ys, pred_ys, average='binary'))
    # # (0.4071969696969697, 0.9847328244274809, 0.5761500669941938, None)

    # Benign & Harmful performance
    train_examples = load_examples("./outputs/iris_l2_wildguard_layer_19_v2/WildGuardMixDataset/train/4000_prompts.jsonl")
    train_xs, train_ys = ensemble_examples(train_examples)
    
    test_examples = load_examples("./outputs/iris_l2_wildguard_layer_19_v2/ORBenchDataset/test/harmful_prompts.jsonl")
    test_examples += load_examples("./outputs/iris_l2_wildguard_layer_19_v2/ORBenchDataset/test/hard_benign_prompts.jsonl")
    test_xs, test_ys = ensemble_examples(test_examples)

    model = DecisionTreeClassifier(random_state=42)
    model.fit(train_xs, train_ys)
    pred_ys = model.predict(test_xs)
    print("Benign & Harmful performance:")
    print(precision_recall_fscore_support(test_ys, pred_ys, average='binary'))
    # 

    # # Ensemble performance
    # train_examples_1 = load_examples("./outputs/iris_l2_wildguard_layer_19_benign_only_v2/WildGuardMixDataset/train/4000_prompts.jsonl")
    # train_examples_2 = load_examples("./outputs/iris_l2_wildguard_layer_19_b0h1_v2/WildGuardMixDataset/train/4000_prompts.jsonl")
    # train_xs, train_ys = ensemble_examples(train_examples_1, train_examples_2)
    
    # test_examples_1 = load_examples("./outputs/iris_l2_wildguard_layer_19_benign_only_v2/ORBenchDataset/test/harmful_prompts.jsonl")
    # test_examples_1 += load_examples("./outputs/iris_l2_wildguard_layer_19_benign_only_v2/ORBenchDataset/test/hard_benign_prompts.jsonl")
    # test_examples_2 = load_examples("./outputs/iris_l2_wildguard_layer_19_b0h1_v2/ORBenchDataset/test/harmful_prompts.jsonl")
    # test_examples_2 += load_examples("./outputs/iris_l2_wildguard_layer_19_b0h1_v2/ORBenchDataset/test/hard_benign_prompts.jsonl")
    # test_xs, test_ys = ensemble_examples(test_examples_1, test_examples_2)

    # model = DecisionTreeClassifier(random_state=42)
    # model.fit(train_xs, train_ys)
    # pred_ys = model.predict(test_xs)
    # print("Ensemble performance:")
    # print(precision_recall_fscore_support(test_ys, pred_ys, average='binary'))
    # # (0.4173633440514469, 0.9908396946564886, 0.5873303167420815, None)