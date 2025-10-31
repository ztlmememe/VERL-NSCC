#!/usr/bin/env bash

export RAY_ADDRESS="http://10.168.4.155:8265"
# Make the project workspace the working directory Ray ships to workers
export WORKING_DIR="/workspace"
export HOME="/root"

# Point Ray to the runtime env file
export RUNTIME_ENV="/workspace/recipe/dapo/runtime_env.yaml"


echo "==================== [DEBUG] ENVIRONMENT CHECK ===================="
echo "Current hostname: x1000c1s3b0n1"
echo "Current working directory: /home/users/ntu/tianle00/scratch/workshop_demo/VERL-NSCC"
echo "Container HOME: $HOME"
echo "Mounted /root contents:"
ls -al /root || true

# 检查模型目录存在性
if [ -d "/root/verl/models" ]; then
    echo "✅ Found /root/verl/models"
    echo "Contents:"
    ls -al /root/verl/models
else
    echo "❌ ERROR: /root/verl/models not found!"
    echo "Please check that the host path is correctly bound in SARGS:"
    echo "   --bind /home/users/ntu/tianle00/scratch/workshop_demo/tmp_cache:/root"
    exit 1
fi

# 检查模型权重是否存在
if [ -f "/root/verl/models/models--Qwen--Qwen2.5-0.5B-Instruct/config.json" ]; then
    echo "✅ Model config.json found"
else
    echo "⚠️  WARNING: Model config.json not found in /root/verl/models/"
fi

echo "=================================================================="


bash recipe/dapo/run_dapo_qwen2.5_0.5b_2gpu_2nodes.sh
