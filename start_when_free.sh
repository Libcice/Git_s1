#!/bin/bash

# 配置
NEED_GPU_MB=24000  # 需要的显存(MB)，根据你的需求调整
CHECK_INTERVAL=1800  # 检查间隔(秒)

# 训练命令 - 你可以修改这里的参数
TRAIN_CMD="cd /home/liwenlei/pymarl-rlc-main && 
python3 src/main.py --config=qmix_history_token_belief --env-config=sc2 with env_args.map_name=corridor t_max=5010000 epsilon_anneal_time=100000 use_tensorboard=True history_steps=1 "

# 日志文件
LOG_FILE="/tmp/wait_and_train_$(date +%Y%m%d_%H%M%S).log"

echo "========================================" | tee -a "$LOG_FILE"
echo "显存监控启动" | tee -a "$LOG_FILE"
echo "需要显存: ${NEED_GPU_MB}MB" | tee -a "$LOG_FILE"
echo "检查间隔: ${CHECK_INTERVAL}秒" | tee -a "$LOG_FILE"
echo "训练命令: ${TRAIN_CMD}" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

while true; do
    # 获取当前显存使用
    USED_MB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    TOTAL_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    FREE_MB=$((TOTAL_MB - USED_MB))

    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$TIMESTAMP] 显存: 已用 ${USED_MB}MB / 总共 ${TOTAL_MB}MB, 剩余 ${FREE_MB}MB" | tee -a "$LOG_FILE"

    if [ "$FREE_MB" -gt "$NEED_GPU_MB" ]; then
        echo "[$TIMESTAMP] 显存足够！开始训练..." | tee -a "$LOG_FILE"
        echo "========================================" | tee -a "$LOG_FILE"
        eval "$TRAIN_CMD" 2>&1 | tee -a "$LOG_FILE"
        echo "========================================" | tee -a "$LOG_FILE"
        echo "[$TIMESTAMP] 训练结束" | tee -a "$LOG_FILE"
        break
    fi

    sleep $CHECK_INTERVAL
done
# # 查看日志
# tail -f /tmp/wait_and_train_*.log