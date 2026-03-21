#!/bin/bash

# 登录 WandB

# ## qiulin13145@gmail.com
# python -m wandb login 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
# 471610515@qq.com
python -m wandb login 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
# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1
export WANDB_PROJECT="60M_model" 

# ================= 配置区域 =================
MODEL_CONFIG="configs/llama_60m.json"
DATASET_PATH="/data/datasets/c4/en"
WORKERS=8

# 训练超参数
BATCH_SIZE=1
TOTAL_BATCH_SIZE=2
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
RUN_NAME_BASE="60M_RL_"
MODE='train'
ROUND=256
RL_BASE_SCHEDULER="cosine"

# Meta-Training 配置
TOTAL_RL_EPOCHS=1
BASE_SEED=128

# [修改] Action Scale 课程学习配置 - 限制最大值为 2.0
MIN_ACTION_SCALE=1.0
MAX_ACTION_SCALE=2.0 
# ===========================================

mkdir -p $RL_SAVE_BASE_DIR

# ===========================================
# 断点续训检测逻辑
# ===========================================
START_EPOCH=1
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
echo "Resuming Training from Epoch: $START_EPOCH"
echo "----------------------------------------------------------"
# ===========================================


for (( epoch=START_EPOCH; epoch<=TOTAL_RL_EPOCHS; epoch++ ))
do
    echo "=========================================================="
    echo "Starting Meta-Training Epoch: $epoch / $TOTAL_RL_EPOCHS"
    echo "=========================================================="

    # 1. 动态设置种子
    CURRENT_SEED=$((BASE_SEED + epoch))
    
    # 2. 动态计算 Action Scale
    CURRENT_ACTION_SCALE=$(python -c "
min_s = $MIN_ACTION_SCALE
max_s = $MAX_ACTION_SCALE
curr_e = $epoch
total_e = $TOTAL_RL_EPOCHS
ramp_up_epochs = 20 

if curr_e >= ramp_up_epochs:
    print(max_s) 
else:
    ratio = (curr_e - 1) / (ramp_up_epochs - 1)
    scale = min_s + (max_s - min_s) * ratio
    print(f'{scale:.4f}')
")

    echo "Using Seed: $CURRENT_SEED"
    echo "Curriculum Action Scale: $CURRENT_ACTION_SCALE (Range: $MIN_ACTION_SCALE -> $MAX_ACTION_SCALE)"

    # 3. 确定 Agent 加载路径
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

    # 4. 启动训练
    while true; do
        mkdir -p $CURRENT_SAVE_DIR
        
         # 调试：打印一下当前目录下有没有 crash 文件
        ls -l ${CURRENT_SAVE_DIR}/agent_latest_crash.pth 2>/dev/null

         # 如果发现了崩溃恢复文件，强制覆盖加载路径
        if [ -f "${CURRENT_SAVE_DIR}/agent_latest_crash.pth" ]; then
            echo "⚠️ Found crash recovery agent, overriding load path..."
            AGENT_LOAD_ARG="--rl_agent_load_path ${CURRENT_SAVE_DIR}/agent_latest_crash.pth"
        fi
        
        # 4. 启动训练
        torchrun --standalone --nproc_per_node=2 torchrun_main_DDP.py \
            --model_name $CURRENT_RUN_NAME \
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
            --save_dir $CURRENT_SAVE_DIR \
            --name $CURRENT_RUN_NAME \
            --wandb_project_name $WANDB_PROJECT \
            --dataset_path $DATASET_PATH \
            --dtype $DTYPE \
            --workers $WORKERS \
            --peft_model full-rank \
            --rank 0 \
            --lora_alpha 32 \
            --rl_agent_lr $RL_AGENT_LR \
            --rl_history_len $RL_HISTORY_LEN \
            --rl_agent_save_dir $RL_SAVE_BASE_DIR \
            --rl_epoch $epoch \
            --seed $CURRENT_SEED \
            --rl_mode $MODE \
            --action_scale $CURRENT_ACTION_SCALE \
            --rl_round $ROUND \
            --rl_base_scheduler $RL_BASE_SCHEDULER \
            $AGENT_LOAD_ARG

        # 检查退出码
        EXIT_CODE=$?

        if [ $EXIT_CODE -eq 0 ]; then
            echo "✅ Epoch $epoch completed successfully."
            break  # 成功了！跳出 while 循环，继续外层的 for 循环
        else
            echo "❌ Training failed at Epoch $epoch with exit code $EXIT_CODE (Likely Loss Spike)."
            echo "🔄 Restarting Epoch $epoch in 10 seconds..."
            
            # 可选：如果你希望重试时改变随机种子以避开特定的坏数据，可以在这里修改 CURRENT_SEED
            # CURRENT_SEED=$((CURRENT_SEED + 1000))
            
            sleep 10
            # 不 break，继续 while 循环 -> 重新执行 torchrun
            # 因为没有保存坏的 checkpoint，它会重新加载上一轮正常的 agent
        fi
    done
    
    # 验证文件
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

echo "Meta-Training Completed Successfully!"