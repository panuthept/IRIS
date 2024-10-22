# echo "Running IRIS on WildGuard with layer 17 config"
# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 scripts/iris_wildguard.py \
# --iris_config ./data/iris_configs/benign_only_configs/layer_17.json \
# --model_name allenai/wildguard \
# --train_eval_split 0.9 \
# --max_seq_length 2048 \
# --batch_size 1 \
# --gradient_accumulation_steps 32 \
# --epochs 2 \
# --eval_steps 60 \
# --save_total_limit 100 \
# --output_dir ./finetuned_models/iris_wildguard_layer_17 \
# --use_lora \
# --lora_rank 128

echo "Running IRIS on WildGuard with layer 18 config"
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 scripts/iris_wildguard.py \
--iris_config ./data/iris_configs/benign_only_configs/layer_18.json \
--model_name allenai/wildguard \
--train_eval_split 0.9 \
--max_seq_length 2048 \
--batch_size 1 \
--gradient_accumulation_steps 32 \
--epochs 2 \
--eval_steps 60 \
--save_total_limit 100 \
--output_dir ./finetuned_models/iris_wildguard_layer_18 \
--use_lora \
--lora_rank 128

echo "Running IRIS on WildGuard with layer 19 config"
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 scripts/iris_wildguard.py \
--iris_config ./data/iris_configs/benign_only_configs/layer_19.json \
--model_name allenai/wildguard \
--train_eval_split 0.9 \
--max_seq_length 2048 \
--batch_size 1 \
--gradient_accumulation_steps 32 \
--epochs 2 \
--eval_steps 60 \
--save_total_limit 100 \
--output_dir ./finetuned_models/iris_wildguard_layer_19 \
--use_lora \
--lora_rank 128

echo "Running IRIS on WildGuard with layer 20 config"
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 scripts/iris_wildguard.py \
--iris_config ./data/iris_configs/benign_only_configs/layer_20.json \
--model_name allenai/wildguard \
--train_eval_split 0.9 \
--max_seq_length 2048 \
--batch_size 1 \
--gradient_accumulation_steps 32 \
--epochs 2 \
--eval_steps 60 \
--save_total_limit 100 \
--output_dir ./finetuned_models/iris_wildguard_layer_20 \
--use_lora \
--lora_rank 128

echo "Running IRIS on WildGuard with layer 21 config"
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 scripts/iris_wildguard.py \
--iris_config ./data/iris_configs/benign_only_configs/layer_21.json \
--model_name allenai/wildguard \
--train_eval_split 0.9 \
--max_seq_length 2048 \
--batch_size 1 \
--gradient_accumulation_steps 32 \
--epochs 2 \
--eval_steps 60 \
--save_total_limit 100 \
--output_dir ./finetuned_models/iris_wildguard_layer_21 \
--use_lora \
--lora_rank 128

echo "Running IRIS on WildGuard with layer 22 config"
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 scripts/iris_wildguard.py \
--iris_config ./data/iris_configs/benign_only_configs/layer_22.json \
--model_name allenai/wildguard \
--train_eval_split 0.9 \
--max_seq_length 2048 \
--batch_size 1 \
--gradient_accumulation_steps 32 \
--epochs 2 \
--eval_steps 60 \
--save_total_limit 100 \
--output_dir ./finetuned_models/iris_wildguard_layer_22 \
--use_lora \
--lora_rank 128

echo "Running IRIS on WildGuard with layer 23 config"
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 scripts/iris_wildguard.py \
--iris_config ./data/iris_configs/benign_only_configs/layer_23.json \
--model_name allenai/wildguard \
--train_eval_split 0.9 \
--max_seq_length 2048 \
--batch_size 1 \
--gradient_accumulation_steps 32 \
--epochs 2 \
--eval_steps 60 \
--save_total_limit 100 \
--output_dir ./finetuned_models/iris_wildguard_layer_23 \
--use_lora \
--lora_rank 128

echo "Running IRIS on WildGuard with layer 24 config"
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 scripts/iris_wildguard.py \
--iris_config ./data/iris_configs/benign_only_configs/layer_24.json \
--model_name allenai/wildguard \
--train_eval_split 0.9 \
--max_seq_length 2048 \
--batch_size 1 \
--gradient_accumulation_steps 32 \
--epochs 2 \
--eval_steps 60 \
--save_total_limit 100 \
--output_dir ./finetuned_models/iris_wildguard_layer_24 \
--use_lora \
--lora_rank 128

echo "Running IRIS on WildGuard with layer 25 config"
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 scripts/iris_wildguard.py \
--iris_config ./data/iris_configs/benign_only_configs/layer_25.json \
--model_name allenai/wildguard \
--train_eval_split 0.9 \
--max_seq_length 2048 \
--batch_size 1 \
--gradient_accumulation_steps 32 \
--epochs 2 \
--eval_steps 60 \
--save_total_limit 100 \
--output_dir ./finetuned_models/iris_wildguard_layer_25 \
--use_lora \
--lora_rank 128

echo "Running IRIS on WildGuard with layer 26 config"
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 scripts/iris_wildguard.py \
--iris_config ./data/iris_configs/benign_only_configs/layer_26.json \
--model_name allenai/wildguard \
--train_eval_split 0.9 \
--max_seq_length 2048 \
--batch_size 1 \
--gradient_accumulation_steps 32 \
--epochs 2 \
--eval_steps 60 \
--save_total_limit 100 \
--output_dir ./finetuned_models/iris_wildguard_layer_26 \
--use_lora \
--lora_rank 128

echo "Running IRIS on WildGuard with layer 27 config"
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 scripts/iris_wildguard.py \
--iris_config ./data/iris_configs/benign_only_configs/layer_27.json \
--model_name allenai/wildguard \
--train_eval_split 0.9 \
--max_seq_length 2048 \
--batch_size 1 \
--gradient_accumulation_steps 32 \
--epochs 2 \
--eval_steps 60 \
--save_total_limit 100 \
--output_dir ./finetuned_models/iris_wildguard_layer_27 \
--use_lora \
--lora_rank 128

echo "Running IRIS on WildGuard with layer 28 config"
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 scripts/iris_wildguard.py \
--iris_config ./data/iris_configs/benign_only_configs/layer_28.json \
--model_name allenai/wildguard \
--train_eval_split 0.9 \
--max_seq_length 2048 \
--batch_size 1 \
--gradient_accumulation_steps 32 \
--epochs 2 \
--eval_steps 60 \
--save_total_limit 100 \
--output_dir ./finetuned_models/iris_wildguard_layer_28 \
--use_lora \
--lora_rank 128