#!/bin/bash

# 登录 WandB
# python -m wandb login ...

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_PROJECT="RL_Inference_130m_Application" 

# ================= 配置区域 =================
# 1. 模型配置改为 130M
MODEL_CONFIG="configs/llama_130m.json"
DATASET_PATH="/data/datasets/c4/en"
WORKERS=8
MODE="eval"

# 2. 训练参数 (130M 规模)
BATCH_SIZE=64      
TOTAL_BATCH_SIZE=512
LR=0.001  
WARMUP_STEPS=2600
MAX_STEPS=26000     
WEIGHT_DECAY=0.0
DTYPE="bfloat16"

# 保存与评估频率
SAVE_EVERY=1000
EVAL_EVERY=1000

# RL 相关配置
RL_AGENT_LR=0.0 # 推理模式下这个不起作用，但为了参数完整性保留
RL_HISTORY_LEN=5
RL_SAVE_BASE_DIR="checkpoints/rl_inference_130m"
RUN_NAME="llama_130m_rl_scheduled_2.6B"

# 3. 指定训练好的 Agent 路径 (请修改为你最好的那个 checkpoint)
BEST_AGENT_PATH="/home/sql/PEFT/checkpoints/rl_agents_meta_run/agent_epoch8_final.pth"

# 检查 Agent 是否存在
if [ ! -f "$BEST_AGENT_PATH" ]; then
    echo "CRITICAL ERROR: Best agent checkpoint not found at:"
    echo "$BEST_AGENT_PATH"
    exit 1
fi

SEED=100
# ===========================================

echo "=========================================================="
echo "Starting 130M Model Training with Pre-trained RL Scheduler"
echo "Agent: $BEST_AGENT_PATH"
echo "=========================================================="

mkdir -p $RL_SAVE_BASE_DIR

# 启动训练
torchrun --standalone --nproc_per_node=8 torchrun_main_DDP.py \
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
    --save_dir "checkpoints/${RUN_NAME}" \
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
    --rl_agent_save_dir $RL_SAVE_BASE_DIR \
    --rl_epoch 1 \
    --seed $SEED \
    --rl_agent_load_path $BEST_AGENT_PATH \
    --rl_mode $MODE  

if [ $? -eq 0 ]; then
    echo "Training finished successfully."
else
    echo "Training failed."
    exit 1
fi