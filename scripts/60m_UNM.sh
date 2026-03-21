python -m wandb login 585b4959ccb98b1ea4d6466883052012b2c9cca8
export WANDB_PROJECT="draft2"
# export WANDB_PROJECT="LoHi"
# export WANDB_PROJECT="splora_SVD_restart"
# export WANDB_PROJECT="opt_project"
# export WANDB_PROJECT="LoHi_debug"

torchrun --standalone --nproc_per_node 2 torchrun_main_DDP.py \
    --model_name UNM_test \
    --model_config configs/llama_60m.json \
    --lr 0.002 \
    --peft_model full-rank \
    --optimizer unbalanced_momentum_v1 \
    --batch_size 2 \
    --total_batch_size 512 \
    --max_length 512\
    --num_training_steps 11000 \
    --warmup_steps 1100 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --dataset_path /data/datasets/c4_en \
    --seed 52 \
