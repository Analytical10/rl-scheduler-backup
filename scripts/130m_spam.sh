python -m wandb login 585b4959ccb98b1ea4d6466883052012b2c9cca8
# export WANDB_PROJECT="draft2"
export WANDB_PROJECT="RL_Inference_130m_Application" 
# export WANDB_PROJECT="splora_SVD_restart"
# export WANDB_PROJECT="opt_project"
# export WANDB_PROJECT="LoHi_debug"

torchrun --standalone --nproc_per_node 8 torchrun_main_DDP.py \
    --model_name spam_bf16_130m_001 \
    --model_config configs/llama_130m.json \
    --lr 0.001 \
    --peft_model full-rank \
    --optimizer stable_spam_adamw \
    --batch_size 64 \
    --total_batch_size 512 \
    --num_training_steps 26000 \
    --warmup_steps 2600 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --dataset_path /data/datasets/c4/en \
    --seed 52 \

