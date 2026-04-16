#!/bin/bash

# 登录 WandB (保留你的设置)
python -m wandb login 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

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1
export WANDB_PROJECT="130M_Llama_InverseSqrt_Standard" 

# ================= 配置区域 =================
MODEL_CONFIG="configs/llama_130m.json" # 确保该文件存在
DATASET_PATH="/data/datasets/c4/en"
WORKERS=8

# 训练超参数 (针对 130M 模型优化)
BATCH_SIZE=128      # 减小了一点点以保证稳定性
TOTAL_BATCH_SIZE=512
LR=0.001              # 峰值学习率
WARMUP_STEPS=2600     # Inverse_Sqrt 建议 Warmup 稍微长一点
MAX_STEPS=26000      # 逆平方根通常用于长时训练，步数可以设大一点
WEIGHT_DECAY=0.1
DTYPE="bfloat16"
MIN_LR_RATIO=0.01     # 设定 1% 的底线学习率，防止后期完全不动

# 保存与评估频率
SAVE_EVERY=5000
EVAL_EVERY=1000

# 路径设置
RUN_NAME="130M_AdamW_InvSqrt_Run1"
SAVE_DIR="checkpoints/${RUN_NAME}"
mkdir -p $SAVE_DIR

# ===========================================
# 启动训练 (去掉了 RL 相关的复杂循环，回归标准训练)
# ===========================================

echo "----------------------------------------------------------"
echo "Starting Training: $RUN_NAME"
echo "Optimizer: AdamW | Scheduler: Inverse_Sqrt"
echo "Peak LR: $LR | Warmup: $WARMUP_STEPS | Min Ratio: $MIN_LR_RATIO"
echo "----------------------------------------------------------"

# 注意：这里去掉了 --optimizer rl_adamw，改回标准的 adamw
# 确保你已经按照我之前的建议修改了 training_utils.py 里的 get_scheculer

torchrun --standalone --nproc_per_node=2 torchrun_main_DDP.py \
    --model_name $RUN_NAME \
    --model_config $MODEL_CONFIG \
    --batch_size $BATCH_SIZE \
    --total_batch_size $TOTAL_BATCH_SIZE \
    --lr $LR \
    --warmup_steps $WARMUP_STEPS \
    --num_training_steps $MAX_STEPS \
    --optimizer "adamw" \
    --scheduler "inverse_sqrt" \
    --min_lr_ratio $MIN_LR_RATIO \
    --weight_decay $WEIGHT_DECAY \
    --save_every $SAVE_EVERY \
    --eval_every $EVAL_EVERY \
    --save_dir $SAVE_DIR \
    --wandb_project_name $WANDB_PROJECT \
    --dataset_path $DATASET_PATH \
    --dtype $DTYPE \
    --workers $WORKERS \
    --peft_model full-rank \
    --seed 42

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully."
else
    echo "❌ Training failed with exit code $EXIT_CODE."
    exit $EXIT_CODE
fi