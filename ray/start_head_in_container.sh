#!/usr/bin/env bash

echo "Running start_head_in_container.sh script that was created at 2025-10-31_16-16-17"

export WORKING_DIR="/workspace"
export HOME="/root"
# Point Ray to the runtime env file
export RUNTIME_ENV="/workspace/recipe/dapo/runtime_env.yaml"

ray stop --force

ray start --head     --node-ip-address='10.168.12.185'     --port='6379'     --dashboard-host=0.0.0.0 --dashboard-port='8265'     --num-gpus '0'     --disable-usage-stats 

echo "SLEEPING FOR 43200 s, to keep Ray cluster up"
sleep 43200

