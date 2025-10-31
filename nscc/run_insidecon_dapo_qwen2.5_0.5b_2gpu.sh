#!/usr/bin/env bash
# set -xeuo pipefail

# Ray dashboard/head address
export RAY_ADDRESS="http://${RAY_IP:-127.0.0.1}:8265"

# Make the project root the working directory Ray ships to workers
export WORKING_DIR="$PWD"

export HOME="/root"

# Point Ray to the runtime env file
export RUNTIME_ENV="$WORKING_DIR/recipe/dapo/runtime_env.yaml"

echo ls inside containter:
ls 
echo pwd inside containter:
pwd
echo "[PBS] HOME: ${HOME}  RUNTIME_ENV:${RUNTIME_ENV}  WORKING_DIR:${WORKING_DIR}"

ray stop --force || true
ray start --head

bash recipe/dapo/run_dapo_qwen2.5_0.5b_2gpu.sh