#!/bin/bash
# Qwen2Moe-1B pretraining with AdamW, using HuggingFace online Pile dataset.
# Adjust --nproc_per_node to match your GPU count.

python -m wandb login 585b4959ccb98b1ea4d6466883052012b2c9cca8
export WANDB_PROJECT="qwen2moe_pretraining"

BATCH_SIZE=64
TOTAL_BATCH_SIZE=512
LR=0.001
WARMUP_STEPS=10000
NUM_TRAINING_STEPS=100000
WEIGHT_DECAY=0.1

torchrun --standalone --nproc_per_node 8 torchrun_main_DDP_qwen2moe.py \
    --model_name        qwen2moe_1b_adam_baseline \
    --moe_config        qwen2-1B \
    --num_experts       32 \
    --num_experts_per_tok 4 \
    --router_aux_loss_coef 0.01 \
    --router_z_loss_coef   1e-3 \
    --hf_dataset \
    --tokenizer_path    ./llama2tokenizer \
    --optimizer         adamw \
    --lr                3e-4 \
    --weight_decay      $WEIGHT_DECAY \
    --grad_clipping     0.0 \
    --scheduler         cosine \
    --min_lr_ratio      0.1 \
    --warmup_steps      $WARMUP_STEPS \
    --num_training_steps $NUM_TRAINING_STEPS \
    --batch_size        $BATCH_SIZE \
    --total_batch_size  $TOTAL_BATCH_SIZE \
    --dtype             bfloat16 \
    --eval_every        1000 \
    --save_every        1000000 \
    --seed              42 \
    --workers           8 \
    --wandb_project_name "$WANDB_PROJECT" \
