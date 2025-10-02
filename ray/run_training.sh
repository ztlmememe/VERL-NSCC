#!/usr/bin/env bash

export RAY_ADDRESS="http://10.168.4.20:8265"
# Make the project workspace the working directory Ray ships to workers
export WORKING_DIR="/workspace"
export HOME="/root"

# Point Ray to the runtime env file
export RUNTIME_ENV="/workspace/recipe/dapo/runtime_env.yaml"

bash recipe/dapo/run_dapo_qwen3_4b_2nodes4A100.sh
