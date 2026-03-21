
# 登录 WandB
python -m wandb login 585b4959ccb98b1ea4d6466883052012b2c9cca8

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_PROJECT="random_adam" 

# ================= 配置区域 =================
MODEL_CONFIG="configs/llama_60m.json"
DATASET_PATH="/data/datasets/c4/en"
WORKERS=8

# 训练超参数
BATCH_SIZE=64
TOTAL_BATCH_SIZE=512
LR=0.003
WARMUP_STEPS=1100
MAX_STEPS=11000
WEIGHT_DECAY=0.1
DTYPE="bfloat16"

# 保存与评估频率
SAVE_EVERY=1000
EVAL_EVERY=1000

RUN_NAME_BASE="Random_adamw"

# Meta-Training 配置
TOTAL_RL_EPOCHS=30
BASE_SEED=128


torchrun --standalone --nproc_per_node=8 torchrun_main_DDP.py \
    --model_config $MODEL_CONFIG \
    --dataset_path $DATASET_PATH \
    --workers $WORKERS \
    --batch_size $BATCH_SIZE \
    --total_batch_size $TOTAL_BATCH_SIZE \
    --lr $LR \
    --warmup_steps $WARMUP_STEPS \
    --num_training_steps $MAX_STEPS \
    --weight_decay $WEIGHT_DECAY \
    --dtype $DTYPE \
    --save_every $SAVE_EVERY \
    --eval_every $EVAL_EVERY \
    --name "$RUN_NAME_BASE" \
    --seed $BASE_SEED \
    --rank 0 \
    --optimizer r_adamw \
    --scheduler cosine \
    --min_lr_ratio 0.1 \

