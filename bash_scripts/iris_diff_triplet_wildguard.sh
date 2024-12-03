CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 scripts/iris_diff_triplet_wildguard.py \
--iris_config ./data/iris_diff_triplet_configs/layer_1_to_10.json \
--iris_label_path ./data/saved_directions \
--model_name allenai/wildguard \
--train_eval_split 0.9 \
--max_seq_length 2048 \
--batch_size 1 \
--gradient_accumulation_steps 32 \
--epochs 2 \
--eval_steps 60 \
--save_total_limit 100 \
--output_dir ./finetuned_models/iris_diff_triplet_wildguard_layer_1_to_10 \
--use_lora \
--lora_rank 128

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 scripts/iris_diff_triplet_wildguard.py \
--iris_config ./data/iris_diff_triplet_configs/layer_11_to_18.json \
--iris_label_path ./data/saved_directions \
--model_name allenai/wildguard \
--train_eval_split 0.9 \
--max_seq_length 2048 \
--batch_size 1 \
--gradient_accumulation_steps 32 \
--epochs 2 \
--eval_steps 60 \
--save_total_limit 100 \
--output_dir ./finetuned_models/iris_diff_triplet_wildguard_layer_11_to_18 \
--use_lora \
--lora_rank 128