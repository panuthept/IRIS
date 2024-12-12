layer=7
alpha=0.9

for epoch in {4..10..2}
do
    echo "Finetuning WildGuard-IRIS-L2 on layer $layer with alpha $alpha for $epoch epochs"
    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 scripts/iris_l2_wildguard.py \
    --iris_config ./data/iris_l2_configs/layer_${layer}_alpha${alpha}.json \
    --model_name allenai/wildguard \
    --train_eval_split 0.9 \
    --max_seq_length 2048 \
    --batch_size 1 \
    --gradient_accumulation_steps 32 \
    --epochs $epoch \
    --eval_steps 60 \
    --save_total_limit 100 \
    --output_dir ./finetuned_models/iris_l2_wildguard_layer_${layer}_alpha${alpha}_epoch${epoch}_v2 \
    --use_lora \
    --lora_rank 128
done

# for j in {60..1200..60}
# do
#     rm -r ./finetuned_models/iris_l2_wildguard_layer_${layer}_alpha${alpha}_v2/checkpoint-${j}
# done

# echo "Inference on WildGuard-IRIS-L2 on layer $layer with alpha $alpha"
# CUDA_VISIBLE_DEVICES=0 python scripts/inference_wildguard.py \
# --model_name allenai/wildguard \
# --checkpoint_path ./finetuned_models/iris_l2_wildguard_layer_${layer}_alpha${alpha}_v2/checkpoint-1220 \
# --dataset_name WildGuardMixDataset \
# --dataset_split train \
# --max_samples 4000 \
# --save_activations \
# --save_logits \
# --output_path ./outputs/iris_l2_wildguard_layer_${layer}_alpha${alpha}_v2/WildGuardMixDataset/train/4000_prompts.jsonl

# CUDA_VISIBLE_DEVICES=0 python scripts/inference_wildguard.py \
# --model_name allenai/wildguard \
# --checkpoint_path ./finetuned_models/iris_l2_wildguard_layer_${layer}_alpha${alpha}_v2/checkpoint-1220 \
# --dataset_name ORBenchDataset \
# --dataset_split test \
# --prompt_intention hard_benign \
# --save_activations \
# --save_logits \
# --output_path ./outputs/iris_l2_wildguard_layer_${layer}_alpha${alpha}_v2/ORBenchDataset/test/hard_benign_prompts.jsonl

# CUDA_VISIBLE_DEVICES=0 python scripts/inference_wildguard.py \
# --model_name allenai/wildguard \
# --checkpoint_path ./finetuned_models/iris_l2_wildguard_layer_${layer}_alpha${alpha}_v2/checkpoint-1220 \
# --dataset_name ORBenchDataset \
# --dataset_split test \
# --prompt_intention harmful \
# --save_activations \
# --save_logits \
# --output_path ./outputs/iris_l2_wildguard_layer_${layer}_alpha${alpha}_v2/ORBenchDataset/test/harmful_prompts.jsonl