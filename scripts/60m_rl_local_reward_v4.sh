#!/bin/bash

# 单实验版：对齐 60m_rl_train_debug.sh 的训练超参数和流程，仅额外打开 local reward 配置。

# 登录 WandB（按需启用）
# 471610515@qq.com
python -m wandb login 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

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export WANDB_PROJECT=${WANDB_PROJECT:-60M_model_local_reward_test_v4.3}

# ================= 配置区域 =================
MODEL_CONFIG="configs/llama_60m.json"
DATASET_PATH="/data/datasets/c4/en"
WORKERS=8

# 训练超参数（与 60m_rl_train_debug.sh 对齐）
BATCH_SIZE=256
TOTAL_BATCH_SIZE=512
LR=0.003
WARMUP_STEPS=1100
MAX_STEPS=11000
WEIGHT_DECAY=0.1
DTYPE="bfloat16"

# 保存与评估频率（与 debug 脚本对齐）
SAVE_EVERY=2000
EVAL_EVERY=1000

# RL 相关配置,v4.3 round = 0; v4.2 round = 256
RL_AGENT_LR=3e-4
RL_HISTORY_LEN=5
RL_STATS_WINDOW=20
RL_SAVE_BASE_DIR="checkpoints/rl_agents_meta_run_v4.3"
RUN_NAME_BASE="60M_RL_local_reward_test_v4.3"
MODE='train'
ROUND=0
RL_BASE_SCHEDULER="cosine"

# Meta-Training 配置（保持与 debug 脚本相同）
TOTAL_RL_EPOCHS=7
BASE_SEED=128

# Action Scale 课程学习（与 debug 脚本对齐）
MIN_ACTION_SCALE=1.0
MAX_ACTION_SCALE=2.0

# Local reward 配置（保持 global 原权重，放大 alpha 对冲 1e3 量级差）
RL_LOCAL_REWARD_EMA_GAMMA=0.9
# v4设置
RL_ALPHA_START=20.0
RL_ALPHA_END=12.0
RL_BETA_START=0.1
RL_BETA_END=0.7

RL_REWARD_CLIP=20.0
# ===========================================

mkdir -p "$RL_SAVE_BASE_DIR"

START_EPOCH=1
# 断点续训检测：自动跳过已完成 epoch
for (( i=1; i<=TOTAL_RL_EPOCHS; i++ ))
do
    CHECK_PATH="${RL_SAVE_BASE_DIR}/agent_epoch${i}_final.pth"
    if [ -f "$CHECK_PATH" ]; then
        echo "Found completed checkpoint for Epoch $i. Skipping..."
        START_EPOCH=$((i + 1))
    else
        break
    fi
done

if [ $START_EPOCH -gt $TOTAL_RL_EPOCHS ]; then
    echo "All epochs ($TOTAL_RL_EPOCHS) are already completed!"
    exit 0
fi

echo "----------------------------------------------------------"
echo "Running Local Reward Test from Epoch: $START_EPOCH"
echo "----------------------------------------------------------"

for (( epoch=START_EPOCH; epoch<=TOTAL_RL_EPOCHS; epoch++ ))
do
    echo "=========================================================="
    echo "Starting Local Reward Test Epoch: $epoch / $TOTAL_RL_EPOCHS"
    echo "=========================================================="

    CURRENT_SEED=$((BASE_SEED + epoch))

    CURRENT_ACTION_SCALE=$(python -c "
min_s = $MIN_ACTION_SCALE
max_s = $MAX_ACTION_SCALE
curr_e = $epoch
ramp_up_epochs = 20

if curr_e >= ramp_up_epochs:
    print(max_s)
else:
    ratio = (curr_e - 1) / (ramp_up_epochs - 1)
    scale = min_s + (max_s - min_s) * ratio
    print(f'{scale:.4f}')
")

    CURRENT_RUN_NAME="${RUN_NAME_BASE}_epoch${epoch}"
    CURRENT_SAVE_DIR="checkpoints/${CURRENT_RUN_NAME}"

    # Epoch 间继承：默认加载上一轮 agent，除非是第一轮
    AGENT_LOAD_ARG=""
    if [ $epoch -eq 1 ]; then
        echo "Epoch 1: Initializing fresh RL Agent."
    else
        PREV_EPOCH=$((epoch - 1))
        PREV_AGENT_PATH="${RL_SAVE_BASE_DIR}/agent_epoch${PREV_EPOCH}_final.pth"
        if [ ! -f "$PREV_AGENT_PATH" ]; then
            echo "CRITICAL ERROR: Previous agent checkpoint not found: $PREV_AGENT_PATH"
            exit 1
        fi
        AGENT_LOAD_ARG="--rl_agent_load_path $PREV_AGENT_PATH"
        echo "Epoch $epoch: Loading RL Agent from $PREV_AGENT_PATH"
    fi

    echo "Using Seed: $CURRENT_SEED"
    echo "Curriculum Action Scale: $CURRENT_ACTION_SCALE (Range: $MIN_ACTION_SCALE -> $MAX_ACTION_SCALE)"
    echo "Local reward mix alpha: $RL_ALPHA_START -> $RL_ALPHA_END"
    echo "Local reward mix beta : $RL_BETA_START -> $RL_BETA_END"

    while true; do
        mkdir -p "$CURRENT_SAVE_DIR"

        ls -l "${CURRENT_SAVE_DIR}/agent_latest_crash.pth" 2>/dev/null || true

        # 优先使用 crash 恢复 agent 覆盖默认加载路径
        if [ -f "${CURRENT_SAVE_DIR}/agent_latest_crash.pth" ]; then
            echo "Found crash recovery agent, overriding load path..."
            AGENT_LOAD_ARG="--rl_agent_load_path ${CURRENT_SAVE_DIR}/agent_latest_crash.pth"
        fi

        torchrun --standalone --nproc_per_node=2 torchrun_main_DDP.py \
            --model_name "$CURRENT_RUN_NAME" \
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
            --rl_stats_window "$RL_STATS_WINDOW" \
            --rl_agent_save_dir "$RL_SAVE_BASE_DIR" \
            --rl_epoch "$epoch" \
            --seed "$CURRENT_SEED" \
            --rl_mode "$MODE" \
            --action_scale "$CURRENT_ACTION_SCALE" \
            --rl_round "$ROUND" \
            --rl_base_scheduler "$RL_BASE_SCHEDULER" \
            --rl_local_reward_ema_gamma "$RL_LOCAL_REWARD_EMA_GAMMA" \
            --rl_alpha_start "$RL_ALPHA_START" \
            --rl_alpha_end "$RL_ALPHA_END" \
            --rl_beta_start "$RL_BETA_START" \
            --rl_beta_end "$RL_BETA_END" \
            --rl_reward_clip "$RL_REWARD_CLIP" \
            $AGENT_LOAD_ARG

        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 0 ]; then
            echo "Epoch $epoch completed successfully."
            break
        else
            echo "Training failed at Epoch $epoch with exit code $EXIT_CODE (Likely Loss Spike)."
            echo "Restarting Epoch $epoch in 10 seconds..."
            sleep 10
        fi
    done

    EXPECTED_FINAL_PATH="${RL_SAVE_BASE_DIR}/agent_epoch${epoch}_final.pth"
    if [ ! -f "$EXPECTED_FINAL_PATH" ]; then
        echo "WARNING: Expected output file $EXPECTED_FINAL_PATH was not found!"
        exit 1
    else
        echo "Success: Found output agent $EXPECTED_FINAL_PATH"
    fi

    echo "Epoch $epoch finished."
    echo "----------------------------------------------------------"
    sleep 5
done

echo "Local reward test completed successfully!"
