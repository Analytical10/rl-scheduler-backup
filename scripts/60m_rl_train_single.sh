#!/bin/bash

# 登录 WandB
python -m wandb login 585b4959ccb98b1ea4d6466883052012b2c9cca8

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_PROJECT="test"

# ================= 配置区域 =================
MODEL_CONFIG="configs/llama_60m.json"
DATASET_PATH="/data/datasets/c4/en"
WORKERS=8

# 训练超参数
BATCH_SIZE=1
TOTAL_BATCH_SIZE=8
LR=0.003
WARMUP_STEPS=1100
MAX_STEPS=11000
WEIGHT_DECAY=0.1
DTYPE="bfloat16"

# 保存与评估频率
SAVE_EVERY=1000
EVAL_EVERY=1000

# RL 相关配置
RL_AGENT_LR=3e-4
RL_HISTORY_LEN=5
RL_SAVE_BASE_DIR="checkpoints/rl_agents_meta_run"
RUN_NAME_BASE="rl_adamw_meta_"
MODE="train"

# Meta-Training 配置
TOTAL_RL_EPOCHS=1
BASE_SEED=42

# 固定的 action_scale（你可以根据需要修改这个值）
ACTION_SCALE=1.5
# ===========================================

mkdir -p "$RL_SAVE_BASE_DIR"

echo "=========================================================="
echo "Starting Meta-Training (No resume, fixed action_scale)"
echo "Total Epochs: $TOTAL_RL_EPOCHS"
echo "Fixed action_scale: $ACTION_SCALE"
echo "=========================================================="

for (( epoch=1; epoch<=TOTAL_RL_EPOCHS; epoch++ ))
do
    echo "=========================================================="
    echo "Starting Meta-Training Epoch: $epoch / $TOTAL_RL_EPOCHS"
    echo "=========================================================="

    # 1. 动态设置种子（仍然随 epoch 变化，方便多样性）
    CURRENT_SEED=$((BASE_SEED + epoch))

    echo "Using Seed: $CURRENT_SEED"
    echo "Using Fixed Action Scale: $ACTION_SCALE"

    # 2. Agent 加载路径：
    #    - 第 1 个 epoch 不加载旧 agent
    #    - 后续 epoch 从上一 epoch 的 final checkpoint 加载
    if [ $epoch -eq 1 ]; then
        AGENT_LOAD_ARG=""
        echo "Epoch 1: Initializing fresh RL Agent."
    else
        PREV_EPOCH=$((epoch - 1))
        PREV_AGENT_PATH="${RL_SAVE_BASE_DIR}/agent_epoch${PREV_EPOCH}_final.pth"

        if [ ! -f "$PREV_AGENT_PATH" ]; then
            echo "CRITICAL ERROR: Previous agent checkpoint not found at:"
            echo "$PREV_AGENT_PATH"
            exit 1
        fi

        AGENT_LOAD_ARG="--rl_agent_load_path $PREV_AGENT_PATH"
        echo "Epoch $epoch: Loading RL Agent from $PREV_AGENT_PATH"
    fi

    CURRENT_RUN_NAME="${RUN_NAME_BASE}_epoch${epoch}"
    CURRENT_SAVE_DIR="checkpoints/${CURRENT_RUN_NAME}"
    mkdir -p "$CURRENT_SAVE_DIR"

    # 3. 启动本 epoch 训练（不做 while 重试，不做 crash 恢复）
    torchrun --standalone --nproc_per_node=8 torchrun_main_DDP.py \
        --model_name "sample_only_1.3" \
        --model_config "$MODEL_CONFIG" \
        --batch_size "$BATCH_SIZE" \
        --total_batch_size "$TOTAL_BATCH_SIZE" \
        --lr "$LR" \
        --warmup_steps "$WARMUP_STEPS" \
        --num_training_steps "$MAX_STEPS" \
        --optimizer rl_adamw \
        --scheduler cosine \
        --min_lr_ratio 0.1 \
        --weight_decay "$WEIGHT_DECAY" \
        --save_every "$SAVE_EVERY" \
        --eval_every "$EVAL_EVERY" \
        --save_dir "$CURRENT_SAVE_DIR" \
        --name "$CURRENT_RUN_NAME" \
        --wandb_project_name "$WANDB_PROJECT" \
        --dataset_path "$DATASET_PATH" \
        --dtype "$DTYPE" \
        --workers "$WORKERS" \
        --peft_model full-rank \
        --rank 0 \
        --lora_alpha 32 \
        --rl_agent_lr "$RL_AGENT_LR" \
        --rl_history_len "$RL_HISTORY_LEN" \
        --rl_agent_save_dir "$RL_SAVE_BASE_DIR" \
        --rl_epoch "$epoch" \
        --seed "$CURRENT_SEED" \
        --rl_mode "$MODE" \
        --action_scale "$ACTION_SCALE" \
        $AGENT_LOAD_ARG

    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "❌ Training failed at Epoch $epoch with exit code $EXIT_CODE."
        echo "Stopping meta-training."
        exit $EXIT_CODE
    fi

    # 4. 简单检查本 epoch 输出 agent 是否存在（可选）
    EXPECTED_FINAL_PATH="${RL_SAVE_BASE_DIR}/agent_epoch${epoch}_final.pth"
    if [ ! -f "$EXPECTED_FINAL_PATH" ]; then
        echo "WARNING: Expected output file $EXPECTED_FINAL_PATH was not found!"
        exit 1
    else
        echo "Success: Found output agent $EXPECTED_FINAL_PATH"
    fi

    echo "Epoch $epoch finished."
    echo "----------------------------------------------------------"
    sleep 3
done

echo "Meta-Training Completed Successfully!"