#!/usr/bin/env bash
# set -xeuo pipefail

# Ray dashboard/head address
export RAY_ADDRESS="http://${RAY_IP:-127.0.0.1}:8265"

# Make the project root the working directory Ray ships to workers
export WORKING_DIR="$PWD"

export HOME="/root"

# Point Ray to the runtime env file
export RUNTIME_ENV="$WORKING_DIR/recipe/dapo/runtime_env.yaml"

ray stop --force || true
ray start --head

bash recipe/dapo/run_dapo_qwen3_4b_1gpu_2.sh