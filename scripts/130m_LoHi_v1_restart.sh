python -m wandb login 585b4959ccb98b1ea4d6466883052012b2c9cca8
# export WANDB_PROJECT="draft2"
export WANDB_PROJECT="LoHi_debug"
# export WANDB_PROJECT="splora_SVD_restart"
# export WANDB_PROJECT="opt_project"
# export WANDB_PROJECT="LoHi_debug"

torchrun --standalone --nproc_per_node 2 torchrun_main_DDP.py \
    --model_name LoHi_v1_mxfp4_130m_001_restart \
    --model_config configs/llama_130m.json \
    --lr 0.001 \
    --peft_model lohi_v1 \
    --optimizer adamw \
    --rank 256 \
    --lora_alpha 32 \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 22000 \
    --warmup_steps 2200 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --dataset_path /data/datasets/c4_en \
    --seed 52 \
    --quant_format mxfp4 \
    --restart \
    --cycle_length 1000\
