#!/bin/sh

ray start --head
export RAY_ADDRESS="http://${RAY_IP:-localhost}:8265" # The Ray cluster address to connect to
export WORKING_DIR="${PWD}" # The local directory to package to the Ray cluster
# Set the runtime environment like env vars and pip packages for the Ray cluster in yaml
export RUNTIME_ENV="./recipe/dapo/runtime_env.yaml" # This sets environment variables for the Ray cluster
# bash recipe/dapo/run_dapo_qwen2.5_32b.sh # or other scripts
bash recipe/dapo/run_dapo_qwen3_4b_ntugpuws.sh
