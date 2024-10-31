for model_name in "iris_wildguard_layer_19_freeze"
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