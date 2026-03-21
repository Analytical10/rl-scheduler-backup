#!/bin/bash

# =========================
# WandB
# =========================
python -m wandb login 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
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export WANDB_PROJECT="1B_model_apply_rl"
export WANDB_PROJECT="test"

# =========================
# 基础配置
# =========================
MODEL_CONFIG="configs/llama_1b.json"
DATASET_PATH="/data/datasets/c4/en"
WORKERS=8

# =========================
# 训练超参数
# =========================
BATCH_SIZE=64
TOTAL_BATCH_SIZE=512
LR=0.001
WARMUP_STEPS=10000
MAX_STEPS=100000
WEIGHT_DECAY=0.1
DTYPE="bfloat16"

# =========================
# 保存与评估频率
# =========================
SAVE_EVERY=1000
EVAL_EVERY=1000

# =========================
# RL Agent 配置
# =========================
RL_AGENT_LR=3e-4
RL_HISTORY_LEN=5

# 这里填你训练好的 agent
RL_AGENT_PATH="checkpoints/rl_agents_meta_run/agent_epoch18_final.pth"

# 关键：使用你验证最好的 action_scale
ACTION_SCALE=1.9

# 关键：保持和训练 agent 时一致
ROUND=24

# 关键：如果只是应用，不继续训练 agent，mode 不要用 train
MODE="eval"

RUN_NAME="130m_apply_rl_epoch18_as1.9"
SAVE_DIR="checkpoints/${RUN_NAME}"

mkdir -p "$SAVE_DIR"

echo "=========================================================="
echo "Applying pretrained RL agent to LLM training"
echo "RL_AGENT_PATH = $RL_AGENT_PATH"
echo "ACTION_SCALE  = $ACTION_SCALE"
echo "ROUND         = $ROUND"
echo "MODE          = $MODE"
echo "SAVE_DIR      = $SAVE_DIR"
echo "=========================================================="

torchrun --standalone --nproc_per_node=8 torchrun_main_DDP.py \
    --no_slice \
    --model_name $RUN_NAME \
    --model_config $MODEL_CONFIG \
    --batch_size $BATCH_SIZE \
    --total_batch_size $TOTAL_BATCH_SIZE \
    --lr $LR \
    --warmup_steps $WARMUP_STEPS \
    --num_training_steps $MAX_STEPS \
    --optimizer rl_adamw \
    --scheduler cosine \
    --min_lr_ratio 0.1 \
    --weight_decay $WEIGHT_DECAY \
    --save_every $SAVE_EVERY \
    --eval_every $EVAL_EVERY \
    --save_dir $SAVE_DIR \
    --name $RUN_NAME \
    --wandb_project_name $WANDB_PROJECT \
    --dataset_path $DATASET_PATH \
    --dtype $DTYPE \
    --workers $WORKERS \
    --peft_model full-rank \
    --rank 0 \
    --lora_alpha 32 \
    --rl_agent_lr $RL_AGENT_LR \
    --rl_history_len $RL_HISTORY_LEN \
    --rl_agent_save_dir $SAVE_DIR \
    --rl_epoch 0 \
    --seed 128 \
    --rl_mode $MODE \
    --action_scale $ACTION_SCALE \
    --rl_round $ROUND \
    --rl_agent_load_path $RL_AGENT_PATH \
    