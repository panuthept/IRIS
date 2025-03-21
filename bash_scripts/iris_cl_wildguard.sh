echo "Running WildGuard-IRIS-CL on layer 19 (Benign only)"
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 scripts/iris_cl_wildguard.py \
--iris_config ./data/iris_l2_configs/benign_only_configs/layer_19.json \
--model_name allenai/wildguard \
--train_eval_split 0.9 \
--max_seq_length 2048 \
--batch_size 1 \
--gradient_accumulation_steps 32 \
--epochs 2 \
--eval_steps 60 \
--save_total_limit 100 \
--output_dir ./finetuned_models/iris_cl_wildguard_layer_19_benign_only \
--use_lora \
--lora_rank 128

echo "Running WildGuard-IRIS-CL on layer 19"
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 scripts/iris_cl_wildguard.py \
--iris_config ./data/iris_l2_configs/layer_19.json \
--model_name allenai/wildguard \
--train_eval_split 0.9 \
--max_seq_length 2048 \
--batch_size 1 \
--gradient_accumulation_steps 32 \
--epochs 2 \
--eval_steps 60 \
--save_total_limit 100 \
--output_dir ./finetuned_models/iris_cl_wildguard_layer_19 \
--use_lora \
--lora_rank 128