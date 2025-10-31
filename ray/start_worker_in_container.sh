#!/usr/bin/env bash

# ray stop --force

export WORKING_DIR="/workspace"
export HOME="/root"
# Point Ray to the runtime env file
export RUNTIME_ENV="/workspace/recipe/dapo/runtime_env.yaml"

ray start --address='10.168.4.155:6379'     --num-gpus '4'     --disable-usage-stats 

echo "SLEEPING FOR 10800 s, to keep Ray worker up"
sleep 10800
