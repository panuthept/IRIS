for i in {1..18}
do 
    echo "Finetuning WildGuard-IRIS-L2 on layer $i"
    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 scripts/iris_l2_wildguard.py \
    --iris_config ./data/iris_l2_configs/layer_${i}.json \
    --model_name allenai/wildguard \
    --train_eval_split 0.9 \
    --max_seq_length 2048 \
    --batch_size 1 \
    --gradient_accumulation_steps 32 \
    --epochs 2 \
    --eval_steps 60 \
    --save_total_limit 100 \
    --output_dir ./finetuned_models/iris_l2_wildguard_layer_${i}_v2 \
    --use_lora \
    --lora_rank 128

    for j in 60 120 180 240 300 360 420 480 540 600 660 720 780 840 900 960 1020 1080 1140 1200
    do
        rm -r ./finetuned_models/iris_l2_wildguard_layer_${i}_v2/checkpoint-${j}
    done

    echo "Inference on WildGuard-IRIS-L2 on layer $i"
    CUDA_VISIBLE_DEVICES=0 python scripts/inference_wildguard.py \
    --model_name allenai/wildguard \
    --checkpoint_path ./finetuned_models/iris_l2_wildguard_layer_${i}_v2/checkpoint-1220 \
    --dataset_name WildGuardMixDataset \
    --dataset_split train \
    --max_samples 4000 \
    --save_activations \
    --save_logits \
    --output_path ./outputs/iris_l2_wildguard_layer_${i}_v2/WildGuardMixDataset/train/4000_prompts.jsonl

    CUDA_VISIBLE_DEVICES=0 python scripts/inference_wildguard.py \
    --model_name allenai/wildguard \
    --checkpoint_path ./finetuned_models/iris_l2_wildguard_layer_${i}_v2/checkpoint-1220 \
    --dataset_name ORBenchDataset \
    --dataset_split test \
    --prompt_intention hard_benign \
    --save_activations \
    --save_logits \
    --output_path ./outputs/iris_l2_wildguard_layer_${i}_v2/ORBenchDataset/test/hard_benign_prompts.jsonl

    CUDA_VISIBLE_DEVICES=0 python scripts/inference_wildguard.py \
    --model_name allenai/wildguard \
    --checkpoint_path ./finetuned_models/iris_l2_wildguard_layer_${i}_v2/checkpoint-1220 \
    --dataset_name ORBenchDataset \
    --dataset_split test \
    --prompt_intention harmful \
    --save_activations \
    --save_logits \
    --output_path ./outputs/iris_l2_wildguard_layer_${i}_v2/ORBenchDataset/test/harmful_prompts.jsonl
done