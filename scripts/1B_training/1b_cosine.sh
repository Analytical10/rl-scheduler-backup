python -m wandb login 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
# export WANDB_PROJECT="draft2"
export WANDB_PROJECT="Baseline" 
# export WANDB_PROJECT="splora_SVD_restart"
# export WANDB_PROJECT="opt_project"
# export WANDB_PROJECT="LoHi_debug"

torchrun --standalone --nproc_per_node 8 torchrun_main_DDP.py \
    --model_name adma_cosine_1b_001 \
    --model_config configs/llama_1b.json \
    --lr 0.001 \
    --peft_model full-rank \
    --optimizer adamw \
    --batch_size 64 \
    --total_batch_size 512 \
    --num_training_steps 100000 \
    --warmup_steps 10000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --dataset_path /data/datasets/c4/en \
    --seed 52 \
    --no_slice \

