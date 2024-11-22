# echo "Running WildGuard-IRISL2 on layer 19 (Benign only)"
# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 scripts/iris_l2_wildguard.py \
# --iris_config ./data/iris_l2_configs/benign_only_configs/layer_19.json \
# --model_name allenai/wildguard \
# --train_eval_split 0.9 \
# --max_seq_length 2048 \
# --batch_size 1 \
# --gradient_accumulation_steps 32 \
# --epochs 2 \
# --eval_steps 60 \
# --save_total_limit 100 \
# --output_dir ./finetuned_models/iris_l2_wildguard_layer_19_benign_only_v2 \
# --use_lora \
# --lora_rank 128

# echo "Running WildGuard-IRISL2 on layer 19"
# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 scripts/iris_l2_wildguard.py \
# --iris_config ./data/iris_l2_configs/layer_19.json \
# --model_name allenai/wildguard \
# --train_eval_split 0.9 \
# --max_seq_length 2048 \
# --batch_size 1 \
# --gradient_accumulation_steps 32 \
# --epochs 2 \
# --eval_steps 60 \
# --save_total_limit 100 \
# --output_dir ./finetuned_models/iris_l2_wildguard_layer_19_v2 \
# --use_lora \
# --lora_rank 128

echo "Running WildGuard-IRISL2 on layer 19 (b1h01)"
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 scripts/iris_l2_wildguard.py \
--iris_config ./data/iris_l2_configs/layer_19_b1h01.json \
--model_name allenai/wildguard \
--train_eval_split 0.9 \
--max_seq_length 2048 \
--batch_size 1 \
--gradient_accumulation_steps 32 \
--epochs 2 \
--eval_steps 60 \
--save_total_limit 100 \
--output_dir ./finetuned_models/iris_l2_wildguard_layer_19_b1h01_v2 \
--use_lora \
--lora_rank 128

echo "Running WildGuard-IRISL2 on layer 19 (b1h02)"
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 scripts/iris_l2_wildguard.py \
--iris_config ./data/iris_l2_configs/layer_19_b1h02.json \
--model_name allenai/wildguard \
--train_eval_split 0.9 \
--max_seq_length 2048 \
--batch_size 1 \
--gradient_accumulation_steps 32 \
--epochs 2 \
--eval_steps 60 \
--save_total_limit 100 \
--output_dir ./finetuned_models/iris_l2_wildguard_layer_19_b1h02_v2 \
--use_lora \
--lora_rank 128

echo "Running WildGuard-IRISL2 on layer 19 (b1h05)"
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 scripts/iris_l2_wildguard.py \
--iris_config ./data/iris_l2_configs/layer_19_b1h05.json \
--model_name allenai/wildguard \
--train_eval_split 0.9 \
--max_seq_length 2048 \
--batch_size 1 \
--gradient_accumulation_steps 32 \
--epochs 2 \
--eval_steps 60 \
--save_total_limit 100 \
--output_dir ./finetuned_models/iris_l2_wildguard_layer_19_b1h05_v2 \
--use_lora \
--lora_rank 128