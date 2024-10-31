for model_name in "iris_wildguard_layer_17" "iris_wildguard_layer_18" "iris_wildguard_layer_19" "iris_wildguard_layer_19_r2" "iris_wildguard_layer_19_negative" "iris_wildguard_layer_19_rand_1" "iris_wildguard_layer_19_rand_2" "iris_wildguard_layer_19_rand_3" "iris_wildguard_layer_19_rand_4" "iris_wildguard_layer_19_smooth_001" "iris_wildguard_layer_19_smooth_005" "iris_wildguard_layer_19_smooth_01" "iris_wildguard_layer_19_smooth_02" "iris_wildguard_layer_20" "iris_wildguard_layer_21" "iris_wildguard_layer_22" "iris_wildguard_layer_23" "iris_wildguard_layer_24" "iris_wildguard_layer_25" "iris_wildguard_layer_26" "iris_wildguard_layer_27"
do
echo "Inference for ${model_name} on JailbreakBenchDataset"
CUDA_VISIBLE_DEVICES=0 python scripts/inference_wildguard.py \
--checkpoint_path ./finetuned_models/${model_name}/checkpoint-1220 \
--dataset_name JailbreakBenchDataset \
--dataset_split test \
--save_logits \
--output_path ./outputs/${model_name}/JailbreakBenchDataset/test/vanilla.jsonl

echo "Inference for ${model_name} on AwesomePromptsDataset"
CUDA_VISIBLE_DEVICES=0 python scripts/inference_wildguard.py \
--checkpoint_path ./finetuned_models/${model_name}/checkpoint-1220 \
--dataset_name AwesomePromptsDataset \
--dataset_split test \
--save_logits \
--output_path ./outputs/${model_name}/AwesomePromptsDataset/test/vanilla.jsonl

echo "Inference for ${model_name} on WildGuardMixDataset (vanilla)"
CUDA_VISIBLE_DEVICES=0 python scripts/inference_wildguard.py \
--checkpoint_path ./finetuned_models/${model_name}/checkpoint-1220 \
--dataset_name WildGuardMixDataset \
--dataset_split test \
--attack_engine vanilla \
--save_logits \
--output_path ./outputs/${model_name}/WildGuardMixDataset/test/vanilla.jsonl

echo "Inference for ${model_name} on WildGuardMixDataset (adversarial)"
CUDA_VISIBLE_DEVICES=0 python scripts/inference_wildguard.py \
--checkpoint_path ./finetuned_models/${model_name}/checkpoint-1220 \
--dataset_name WildGuardMixDataset \
--dataset_split test \
--attack_engine adversarial \
--save_logits \
--output_path ./outputs/${model_name}/WildGuardMixDataset/test/adversarial.jsonl
done